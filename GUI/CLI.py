from simple_term_menu import TerminalMenu

class CLI:
    def __init__(self, questions):
        self.questions = questions

    def run(self):
        self.navigate(self.questions)

    def navigate(self, options):
        if isinstance(options, list):
            self.navigate_steps(options)
        else:
            self.navigate_menu(options)

    def navigate_steps(self, steps):
        for step in steps:
            self.navigate_menu(step)
        return

    def navigate_menu(self, options):
        terminal_menu = TerminalMenu(list(options.keys()))
        menu_entry_index = terminal_menu.show()

        if menu_entry_index is None:
            return

        selected_key = list(options.keys())[menu_entry_index]
        selected_value = options[selected_key]

        if callable(selected_value):
            selected_value()
            return
        elif isinstance(selected_value, dict):
            self.navigate(selected_value)
        elif isinstance(selected_value, list):
            self.navigate_steps(selected_value)