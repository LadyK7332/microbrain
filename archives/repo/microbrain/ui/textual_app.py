from textual.app import App, ComposeResult
from textual.widgets import Input, Static, RichLog
from textual.containers import Vertical

class MicroBrainUI(App):
    CSS = """
    #log {
        height: 70%;
    }
    #history {
        height: 20%;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(id="log", highlight=True)
            yield RichLog(id="history")
            yield Input(placeholder="Speak to MicroBrain…", id="input")

    def on_mount(self) -> None:
        self.query_one("#log", RichLog).write("[system] UI online")
        self.set_focus(self.query_one("#input", Input))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        self.query_one("#history", RichLog).write(f"[you] {text}")
        self.query_one("#log", RichLog).write("[system] (input captured)")

if __name__ == "__main__":
    MicroBrainUI().run()
