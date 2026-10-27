"""Build the Streamlit-embedded game from the same source as standalone web UI.

Keeping a second handwritten HTML template caused the Streamlit app to silently
run stale controls and explanations. This helper now inlines index.html, style.css,
connectome_data.json, and app.js so both entry points use the same UI and logic.
"""
from pathlib import Path
import json


def get_inlined_html() -> str:
    web_dir = Path(__file__).resolve().parent
    html = (web_dir / "index.html").read_text(encoding="utf-8")
    css = (web_dir / "style.css").read_text(encoding="utf-8")
    js = (web_dir / "app.js").read_text(encoding="utf-8")
    connectome_json = json.loads((web_dir / "connectome_data.json").read_text(encoding="utf-8"))
    decoder_json = json.loads((web_dir / "flappy_decoder.json").read_text(encoding="utf-8"))
    chemotaxis_decoder_json = json.loads((web_dir / "chemotaxis_decoder.json").read_text(encoding="utf-8"))
    connectome_script = "<script>window.CONNECTOME_DATA = " + json.dumps(connectome_json, separators=(",", ":")) + ";</script>"
    decoder_script = "<script>window.FLAPPY_DECODER = " + json.dumps(decoder_json, separators=(",", ":")) + ";</script>"
    chemotaxis_decoder_script = "<script>window.CHEMOTAXIS_DECODER = " + json.dumps(chemotaxis_decoder_json, separators=(",", ":")) + ";</script>"

    html = html.replace(
        '<link rel="stylesheet" href="style.css">',
        "<style>\n" + css + "\n</style>",
    )
    html = html.replace('<script src="connectome_bundle.js"></script>', connectome_script)
    html = html.replace('<script src="flappy_decoder.js"></script>', decoder_script)
    html = html.replace('<script src="chemotaxis_decoder.js"></script>', chemotaxis_decoder_script)
    html = html.replace('<script src="app.js"></script>', "<script>\n" + js + "\n</script>")
    return html
