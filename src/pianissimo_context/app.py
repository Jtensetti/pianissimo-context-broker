"""Local browser UI. Nothing is sent to a hosted Gradio service."""
from __future__ import annotations

import atexit
import json
import os
from urllib.request import Request, ProxyHandler, build_opener

from .app_runtime import AppRuntime


def input_devices():
    import sounddevice as sd
    devices = sd.query_devices()
    choices = [(device["name"], i) for i, device in enumerate(devices) if device["max_input_channels"] > 0]
    default = sd.default.device[0]
    return choices, default if any(i == default for _, i in choices) else choices[0][1] if choices else None


def ollama_models():
    try:
        from urllib.request import HTTPRedirectHandler
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, *args):
                return None
        with build_opener(ProxyHandler({}), NoRedirect()).open(
            Request("http://127.0.0.1:11434/api/tags"), timeout=1) as response:
            data = response.read(256001)
        if len(data) > 256000:
            return []
        return [item["name"] for item in json.loads(data).get("models", []) if isinstance(item.get("name"), str)]
    except Exception:
        return []


def build_app(runtime: AppRuntime | None = None, *, devices=None, models=None):
    import gradio as gr
    runtime = runtime or AppRuntime()
    if devices is None:
        try:
            devices, default = input_devices()
        except Exception:
            devices, default = [], None
    else:
        default = devices[0][1] if devices else None
    models = ollama_models() if models is None else models
    model_default = os.environ.get("PIANISSIMO_OLLAMA_MODEL") or (models[0] if models else None)
    css = """
    .gradio-container {max-width:1120px !important; margin:auto;}
    #transcript textarea {font-size:16px; line-height:1.7;}
    #context textarea {font-size:15px; line-height:1.6;}
    footer {display:none !important;}
    """
    with gr.Blocks(title="Pianissimo", css=css, analytics_enabled=False) as app:
        gr.Markdown("# Pianissimo")
        editing = gr.State(False)
        displayed_revision = gr.State(-1)
        with gr.Row():
            audio = gr.Dropdown(devices, value=default, label="Ljudkälla", scale=4)
            refresh = gr.Button("Uppdatera", scale=1)
            status = gr.Textbox(value="Redo", label="Status", interactive=False, scale=2)
        initial = gr.Textbox(label="Initial kontext", lines=3, max_length=4000)
        with gr.Accordion("Inställningar", open=False):
            model = gr.Dropdown(models, value=model_default, allow_custom_value=True, label="Språkmodell")
        with gr.Row():
            start = gr.Button("Starta", variant="primary")
            stop = gr.Button("Stoppa", interactive=False)
        with gr.Row():
            transcript = gr.Textbox(label="Transkript", lines=20, interactive=False,
                                    autoscroll=True, elem_id="transcript", scale=3)
            with gr.Column(scale=2):
                context = gr.Textbox(label="Kontext", lines=13, interactive=False,
                                     max_length=4000, elem_id="context")
                with gr.Row():
                    edit = gr.Button("Redigera", interactive=False)
                    save = gr.Button("Spara", variant="primary", visible=False)
                    cancel = gr.Button("Avbryt", visible=False)
                updated = gr.Textbox(label="Uppdaterad", interactive=False)
        message = gr.Textbox(label="Meddelande", interactive=False, visible=False)
        timer = gr.Timer(.5)

        def poll(is_editing, revision):
            s = runtime.snapshot()
            new_context = gr.skip() if is_editing or revision == s.revision else gr.update(value=s.context)
            return (s.transcript, s.status, new_context, revision if is_editing else s.revision,
                    s.updated, gr.update(interactive=not s.running), gr.update(interactive=s.running and s.status != "Stoppar"),
                    gr.update(interactive=s.running and not is_editing),
                    gr.update(value=s.message, visible=bool(s.message)))

        timer.tick(poll, [editing, displayed_revision],
                   [transcript, status, context, displayed_revision, updated, start, stop, edit, message],
                   queue=False, show_progress="hidden")

        def start_recording(device, model_name, background):
            try:
                runtime.start(device, model_name, background)
                return (gr.update(value="", visible=False), False,
                        gr.update(value=background, interactive=False), -1,
                        gr.update(visible=False), gr.update(visible=False))
            except ValueError as exc:
                runtime.notice(str(exc))
                return (gr.update(value=str(exc), visible=True), gr.skip(), gr.skip(),
                        gr.skip(), gr.skip(), gr.skip())

        start.click(start_recording, [audio, model, initial],
                    [message, editing, context, displayed_revision, save, cancel], queue=False)
        stop.click(runtime.stop, queue=False)

        def begin_edit():
            s = runtime.snapshot()
            return True, gr.update(value=s.context, interactive=True), gr.update(visible=True), gr.update(visible=True)

        edit.click(begin_edit, outputs=[editing, context, save, cancel], queue=False)

        def save_edit(text):
            try:
                runtime.update_context(text)
                runtime.notice("")
                s = runtime.snapshot()
                return (False, gr.update(value=s.context, interactive=False), s.revision,
                        gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False))
            except (ValueError, RuntimeError) as exc:
                runtime.notice(str(exc))
                return (True, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(value=str(exc), visible=True))

        save.click(save_edit, context, [editing, context, displayed_revision, save, cancel, message], queue=False)

        def cancel_edit():
            s = runtime.snapshot()
            return False, gr.update(value=s.context, interactive=False), s.revision, gr.update(visible=False), gr.update(visible=False)

        cancel.click(cancel_edit, outputs=[editing, context, displayed_revision, save, cancel], queue=False)

        def refresh_inputs():
            try:
                choices, selected = input_devices()
            except Exception:
                choices, selected = [], None
            names = ollama_models()
            return gr.update(choices=choices, value=selected), gr.update(choices=names)

        refresh.click(refresh_inputs, outputs=[audio, model], queue=False)
    return app, runtime


def main():
    app, runtime = build_app()
    atexit.register(runtime.shutdown)
    app.launch(server_name="127.0.0.1", share=False, inbrowser=True,
               show_api=False, quiet=True)


if __name__ == "__main__":
    main()
