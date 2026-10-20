"""多格式学术文档统一解析：MinerU 主路径（PDF/DOCX/PPTX → Markdown），轻量回退路径。

MinerU 保留标题层级、表格（HTML）与公式（LaTeX），避免通用切分器破坏结构；
不可用（模型未下载/依赖缺失）时回退 pypdf / python-docx / python-pptx。
首次运行自动从 modelscope 下载模型并设 HF_ENDPOINT 镜像加速。

使用：
    from document_parser import parse_document
    md, meta = parse_document("paper.pdf")   # meta 含 source/title/pages/parser
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# 在任何 HuggingFace 相关导入之前设置镜像与模型源，加速 MinerU 模型下载
# 1. HF_ENDPOINT：HuggingFace 镜像（hf-mirror.com），用于 huggingface_hub 下载
# 2. MINERU_MODEL_SOURCE：MinerU 模型下载源，优先使用 modelscope（国内直达），
#    可通过环境变量覆盖为 huggingface 或 local
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if not os.environ.get("MINERU_MODEL_SOURCE"):
    os.environ["MINERU_MODEL_SOURCE"] = "modelscope"

# 支持的文档后缀
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".doc", ".ppt"}

# 语言列表（MinerU 的 p_lang_list 参数），中英文学术文档默认中文+英文
DEFAULT_LANG = "ch"


def _patch_transformers_for_mineru() -> None:
    """为 transformers 5.x 补回 MinerU 2.x 依赖的旧版 API。

    MinerU 2.x 在模型初始化时会从 transformers.pytorch_utils 导入
    find_pruneable_heads_and_indices 和 prune_linear_layer，
    这两个函数在 transformers 5.0 中被移除。
    通过 monkey-patch 在 MinerU 加载前补回，使 MinerU 可以在 5.x 环境下运行。

    安全性：仅在函数不存在时补丁，4.x 环境下不做任何改动。
    """
    try:
        import transformers.pytorch_utils as ptu
    except ImportError:
        return  # 模块不存在，无法补丁，MinerU 会回退到轻量解析器

    if hasattr(ptu, "find_pruneable_heads_and_indices"):
        return  # transformers 4.x，无需补丁

    import torch

    def find_pruneable_heads_and_indices(
        heads, n_heads, head_size, already_pruned_heads=None
    ):
        """找出可剪枝的注意力头及其索引（transformers 4.x 实现）。"""
        if already_pruned_heads is None:
            already_pruned_heads = set()
        heads = set(heads) - already_pruned_heads
        head_indices = torch.arange(0, n_heads * head_size, step=head_size)
        head_indices_to_prune = torch.cat([head_indices[i] for i in heads])
        return heads, head_indices_to_prune

    def prune_linear_layer(layer, index, dim=0):
        """剪枝线性层（transformers 4.x 实现）。"""
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias[index].clone()
            else:
                b = layer.bias[index].clone()
        else:
            b = None
        new_layer = torch.nn.Linear(W.size(1), W.size(0), bias=b is not None)
        new_layer.weight = torch.nn.Parameter(W)
        if b is not None:
            new_layer.bias = torch.nn.Parameter(b)
        new_layer.to(layer.weight.device)
        return new_layer

    ptu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    ptu.prune_linear_layer = prune_linear_layer
    print("[document_parser] 已补回 transformers 5.x 缺失的 pytorch_utils 函数")


def parse_document(file_path: str) -> tuple[str, dict]:
    """解析单个文档为 Markdown。

    参数:
        file_path: 文档绝对或相对路径。
    返回:
        (markdown_content, metadata) 元组。
        metadata 含 source（文件名）、title、pages（页数，PDF 才有）、parser（解析器名）。
    异常:
        后缀不支持或文件不存在时抛出 ValueError。
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"文件不存在：{file_path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的格式 {suffix}，仅支持 {SUPPORTED_SUFFIXES}")

    # 在调用 MinerU 前补回 transformers 5.x 缺失的 API（4.x 环境下自动跳过）
    _patch_transformers_for_mineru()

    # 优先 MinerU，失败回退轻量解析器
    try:
        return _parse_with_mineru(path)
    except Exception as e:
        err_msg = str(e)
        if "ftfy" in err_msg:
            print(
                "[document_parser] MinerU 解析失败：缺少 ftfy 依赖（公式识别模块 Unimernet 需要），"
                "请运行 pip install ftfy 安装后重试。"
            )
        elif "tokenize" in err_msg or "127.0.0.1" in err_msg:
            # 公式识别模块 Unimernet 的 tokenizer 服务连接被拒，
            # 尝试关闭公式识别后重新解析（布局检测+OCR+表格识别仍可用）
            print(
                "[document_parser] MinerU 公式识别模块 tokenize 服务连接失败，"
                "尝试关闭公式识别后重新解析..."
            )
            try:
                return _parse_with_mineru(path, formula_enable=False)
            except Exception as e2:
                print(f"[document_parser] 关闭公式识别后仍失败，回退轻量解析器：{e2}")
        else:
            print(f"[document_parser] MinerU 解析失败，回退轻量解析器：{e}")
        return _parse_with_fallback(path)


# ---------------- MinerU 主解析路径 ----------------

# 文本层过少阈值（有效字符数）：低于此值视为疑似扫描件（无文本层），
# auto 模式提取不到正文，需升级为 OCR 重试
_MIN_TEXT_CHARS = 500


def _looks_like_scanned(md_content: str) -> bool:
    """判断解析结果是否疑似扫描件：有效字符过少则无文本层。"""
    text = re.sub(r"\s+", "", md_content)
    return len(text) < _MIN_TEXT_CHARS


def _parse_with_mineru(
    path: Path, formula_enable: bool = True, parse_method: str = "auto"
) -> tuple[str, dict]:
    """用 MinerU pipeline 后端解析文档为 Markdown。

    参数:
        path: 文件路径。
        formula_enable: 是否启用公式识别（Unimernet 模型）。
            公式识别失败时可在重试中关闭此项。
        parse_method: MinerU 解析方法（auto/text/ocr/txt）。
            auto 模式先按文本层解析；若结果文本过少（疑似扫描件/无文本层），
            自动升级为 ocr 强制整页 OCR 重试一次，保证扫描 PDF 也能入库。
    """
    from mineru.cli.common import do_parse, MakeMode

    suffix = path.suffix.lower()
    stem = path.stem
    file_bytes = path.read_bytes()

    # 临时输出目录，解析完成后读取 md 再清理
    tmp_dir = tempfile.mkdtemp(prefix="mineru_parse_")
    try:
        do_parse(
            output_dir=tmp_dir,
            pdf_file_names=[stem],
            pdf_bytes_list=[file_bytes],
            p_lang_list=[DEFAULT_LANG],
            backend="pipeline",
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode=MakeMode.MM_MD,
        )
        # PDF 输出路径：tmp_dir/stem/auto/stem.md
        # Office 输出路径：tmp_dir/stem/office/stem.md
        md_path = _locate_md(tmp_dir, stem)
        if md_path is None:
            raise RuntimeError(f"MinerU 未生成 Markdown 文件：{path.name}")
        md_content = md_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 扫描件降级：auto 模式解析出的文本过少 -> 强制 OCR 重试一次
    if parse_method == "auto" and _looks_like_scanned(md_content):
        print(
            "[document_parser] auto 模式提取文本过少，疑似扫描件/无文本层，"
            "升级为 OCR 重新解析..."
        )
        return _parse_with_mineru(
            path, formula_enable=formula_enable, parse_method="ocr"
        )

    meta = {
        "source": path.name,
        "title": stem,
        "pages": _count_pdf_pages(path) if suffix == ".pdf" else None,
        "parser": f"mineru-{parse_method}",
    }
    return md_content, meta


def _locate_md(tmp_dir: str, stem: str) -> Optional[Path]:
    """在 MinerU 输出目录中定位生成的 Markdown 文件。"""
    # PDF pipeline: tmp_dir/stem/auto/stem.md
    candidate = Path(tmp_dir) / stem / "auto" / f"{stem}.md"
    if candidate.exists():
        return candidate
    # Office: tmp_dir/stem/office/stem.md
    candidate = Path(tmp_dir) / stem / "office" / f"{stem}.md"
    if candidate.exists():
        return candidate
    # 兜底：递归搜索 .md 文件
    for md in Path(tmp_dir).rglob("*.md"):
        return md
    return None


def _count_pdf_pages(path: Path) -> Optional[int]:
    """统计 PDF 页数（可选，失败返回 None）。"""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(path))
        return len(doc)
    except Exception:
        return None


# ---------------- 轻量回退解析路径 ----------------

def _parse_with_fallback(path: Path) -> tuple[str, dict]:
    """MinerU 不可用时的回退解析：python-docx / python-pptx / pypdf。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        md = _parse_pdf_fallback(path)
    elif suffix in (".docx", ".doc"):
        md = _parse_docx_fallback(path)
    elif suffix in (".pptx", ".ppt"):
        md = _parse_pptx_fallback(path)
    else:
        raise ValueError(f"回退解析器不支持格式：{suffix}")

    meta = {
        "source": path.name,
        "title": path.stem,
        "pages": _count_pdf_pages(path) if suffix == ".pdf" else None,
        "parser": "fallback",
    }
    return md, meta


def _parse_pdf_fallback(path: Path) -> str:
    """用 pypdf 提取 PDF 文本（无结构，仅纯文本拼接）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_text = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages_text.append(f"## 第 {i} 页\n\n{text.strip()}")
    return "\n\n".join(pages_text)


def _parse_docx_fallback(path: Path) -> str:
    """用 python-docx 解析 Word，按段落 + 标题样式转 Markdown。"""
    from docx import Document

    doc = Document(str(path))
    lines = []
    for para in doc.paragraphs:
        style_name = (para.style.name or "").lower()
        text = para.text.strip()
        if not text:
            continue
        # 根据 Word 样式映射到 Markdown 标题
        if "heading 1" in style_name or style_name == "title":
            lines.append(f"# {text}")
        elif "heading 2" in style_name:
            lines.append(f"## {text}")
        elif "heading 3" in style_name:
            lines.append(f"### {text}")
        elif "heading 4" in style_name:
            lines.append(f"#### {text}")
        else:
            lines.append(text)
    # 表格转 Markdown
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            header = rows[0]
            sep = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
            lines.append("")
            lines.append(header)
            lines.append(sep)
            lines.extend(rows[1:])
            lines.append("")
    return "\n\n".join(lines)


def _parse_pptx_fallback(path: Path) -> str:
    """用 python-pptx 解析 PPT，按幻灯片顺序转 Markdown。"""
    from pptx import Presentation

    prs = Presentation(str(path))
    slides_md = []
    for i, slide in enumerate(prs.slides, 1):
        parts = [f"## 幻灯片 {i}"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        parts.append(text)
            if shape.has_table:
                table = shape.table
                rows = []
                for row in table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    rows.append("| " + " | ".join(cells) + " |")
                if rows:
                    n_cols = len(table.columns)
                    sep = "| " + " | ".join(["---"] * n_cols) + " |"
                    parts.append("")
                    parts.append(rows[0])
                    parts.append(sep)
                    parts.extend(rows[1:])
        slides_md.append("\n\n".join(parts))
    return "\n\n---\n\n".join(slides_md)


# ---------------- 批量解析 ----------------

def parse_directory(dir_path: str, suffixes: Optional[set] = None) -> list[tuple[str, dict]]:
    """批量解析目录下所有支持格式的文档。

    参数:
        dir_path: 目录路径。
        suffixes: 限定后缀集合，默认 SUPPORTED_SUFFIXES。
    返回:
        [(markdown, metadata), ...] 列表。
    """
    suffixes = suffixes or SUPPORTED_SUFFIXES
    results = []
    for p in sorted(Path(dir_path).iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes:
            try:
                md, meta = parse_document(str(p))
                results.append((md, meta))
            except Exception as e:
                print(f"[document_parser] 解析失败 {p.name}: {e}")
    return results
