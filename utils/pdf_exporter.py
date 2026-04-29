import os
import markdown
from xhtml2pdf import pisa
from core.logger import get_logger

logger = get_logger("pdf_exporter")

def markdown_to_pdf(markdown_text: str, output_path: str) -> bool:
    """
    将 Markdown 文本转换为 PDF 文件并保存到本地。
    """
    try:
        # 1. Markdown 转 HTML
        html_body = markdown.markdown(
            markdown_text, 
            extensions=['tables', 'fenced_code', 'toc']
        )
        
        # 2. 构造完整 HTML 并注入基础中文字体样式
        # 在 Windows 环境下，使用常见的中文字体如 Microsoft YaHei 或 SimSun
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: a4 portrait;
                    margin: 2cm;
                }}
                body {{
                    font-family: "Microsoft YaHei", "SimHei", "SimSun", sans-serif;
                    line-height: 1.6;
                    font-size: 14px;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #333333;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 1em;
                }}
                th, td {{
                    border: 1px solid #dddddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 4px;
                    overflow-x: auto;
                }}
                code {{
                    font-family: monospace;
                    background-color: #f5f5f5;
                    padding: 2px 4px;
                    border-radius: 3px;
                }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """
        
        # 3. HTML 转 PDF
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                html_content, dest=pdf_file, encoding='utf-8'
            )
            
        if pisa_status.err:
            logger.error(f"PDF 生成发生错误: {pisa_status.err}")
            return False
            
        logger.info(f"PDF 成功导出至: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"PDF 导出失败: {e}")
        return False
