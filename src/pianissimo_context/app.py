"""Local browser UI. Nothing is sent to a hosted Gradio service."""
from __future__ import annotations

import atexit
import json
import os
from urllib.request import Request, ProxyHandler, build_opener

from .app_runtime import AppRuntime
from .llm import RECOMMENDED_MODEL


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
    models = list(ollama_models() if models is None else models)
    model_default = os.environ.get("PIANISSIMO_OLLAMA_MODEL") or (
        RECOMMENDED_MODEL if RECOMMENDED_MODEL in models else models[0] if models else RECOMMENDED_MODEL)
    if model_default not in models:
        models.append(model_default)
    css = """
    .gradio-container {max-width:1280px !important; margin:auto;}
    #transcript textarea {font-size:16px; line-height:1.7;}
    #context textarea, #context-draft textarea {font-size:15px; line-height:1.6;}
    footer {display:none !important;}
    """
    with gr.Blocks(title="Pianissimo", css=css, analytics_enabled=False) as app:
        gr.Markdown("# Pianissimo")
        editing = gr.State(False)
        displayed_revision = gr.State(-1)
        edit_revision = gr.State(-1)
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
            transcript = gr.Textbox(label="Transkript", lines=24, interactive=False,
                                    autoscroll=True, elem_id="transcript", scale=3)
            with gr.Column(scale=2):
                with gr.Column() as context_view:
                    background = gr.Textbox(label="Bakgrund", lines=3, interactive=False)
                    context = gr.Textbox(label="Aktuellt ämne", lines=5, interactive=False, elem_id="context")
                    memory = gr.Dataframe(headers=["Ämne", "Kontext"], datatype=["str", "str"],
                                          type="array", value=[], col_count=(2, "fixed"),
                                          label="Tidigare ämnen", interactive=False)
                # Timer responses never target any of these draft components.
                with gr.Column(visible=False) as context_editor:
                    background_draft = gr.Textbox(label="Bakgrund", lines=3, max_length=4000, elem_id="background-draft")
                    context_draft = gr.Textbox(label="Aktuellt ämne", lines=5, max_length=1000, elem_id="context-draft")
                    memory_draft = gr.Dataframe(headers=["Ämne", "Kontext"], datatype=["str", "str"],
                        type="array", value=[], col_count=(2, "fixed"), label="Tidigare ämnen",
                        interactive=True, elem_id="memory-draft")
                with gr.Row():
                    edit = gr.Button("Redigera", interactive=False)
                    save = gr.Button("Spara", variant="primary", visible=False)
                    cancel = gr.Button("Avbryt", visible=False)
                updated = gr.Textbox(label="Uppdaterad", interactive=False)
        message = gr.Textbox(label="Meddelande", interactive=False, visible=False)
        timer = gr.Timer(.5)

        def poll(is_editing, revision):
            s = runtime.snapshot()
            changed = not is_editing and revision != s.revision
            return (s.transcript, s.status,
                    gr.update(value=s.context) if changed else gr.skip(),
                    gr.update(value=s.background) if changed else gr.skip(),
                    gr.update(value=s.memory) if changed else gr.skip(),
                    revision if is_editing else s.revision, s.updated,
                    gr.update(interactive=not s.running),
                    gr.update(interactive=s.running and s.status != "Stoppar"),
                    gr.update(interactive=s.running and s.status != "Stoppar" and not is_editing),
                    gr.update(value=s.message, visible=bool(s.message)))

        timer.tick(poll, [editing, displayed_revision],
                   [transcript, status, context, background, memory, displayed_revision,
                    updated, start, stop, edit, message], queue=False, show_progress="hidden")

        controls = [editing, context_view, context_editor, edit, save, cancel, message]

        def closed_editor():
            return (False, gr.update(visible=True), gr.update(visible=False), gr.update(visible=True),
                    gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False))

        def start_recording(device, model_name, briefing):
            try:
                runtime.start(device, model_name, briefing)
                return closed_editor()
            except ValueError as exc:
                runtime.notice(str(exc))
                return (*[gr.skip() for _ in range(6)], gr.update(value=str(exc), visible=True))

        start.click(start_recording, [audio, model, initial], controls, queue=False)
        stop.click(runtime.stop, queue=False)

        def begin_edit():
            s = runtime.snapshot()
            return (True, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=True), gr.update(value="", visible=False),
                    s.background, s.context, s.memory, s.revision)

        edit.click(begin_edit, outputs=controls + [background_draft, context_draft, memory_draft, edit_revision], queue=False)

        def save_edit(briefing, current, topics, revision):
            try:
                runtime.update_stack(briefing, current, topics, revision)
                runtime.notice("")
                return closed_editor()
            except (ValueError, RuntimeError) as exc:
                runtime.notice(str(exc))
                return (*[gr.skip() for _ in range(6)], gr.update(value=str(exc), visible=True))

        save.click(save_edit, [background_draft, context_draft, memory_draft, edit_revision], controls, queue=False)

        def cancel_edit():
            runtime.notice("")
            return closed_editor()

        cancel.click(cancel_edit, outputs=controls, queue=False)

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
