class TextBuilder:
    """
    Client-independent helper for future server-side text construction.
    The current web UI performs stabilization in JavaScript.
    """

    def __init__(self):
        self.text = ""

    def add_character(self, character):
        if character:
            self.text += str(character)

    def add_space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""

    def get_text(self):
        return self.text
