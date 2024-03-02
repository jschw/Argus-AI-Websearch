import flet as ft

from argus_src.core import ArgusWebsearch


# Define a class called Chat that is a subclass of ft.UserControl
class Chat(ft.UserControl):

    def build(self):
        # Create a heading text element
        self.heading = ft.Text(value="Argus AI Webresearch", size=24)
        
        # Create a text input field for user prompts
        self.text_input = ft.TextField(hint_text="Enter your prompt", expand=True, multiline=True)
        
        # Create an empty column to hold the output elements
        self.output_column = ft.Column()
        
        # Enable scrolling in the chat interface
        self.scroll = True

        # Init Argus core
        self.search_engine = ArgusWebsearch()

        self.first_cycle = True
        
        # Create the layout of the chat interface using the ft.Column container
        return ft.Column(
            width=800,
            controls=[
                # Add the heading, text input, and submit button in a row
                self.heading,
                ft.Row(
                    controls=[
                        self.text_input,
                        ft.ElevatedButton("Submit", height=60, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=1)), on_click=self.btn_clicked),
                    ],
                ),
                # Add the output column to display the chatbot responses
                self.output_column,
            ],
        )
    
    def btn_clicked(self, event):
        # Send the user input to the Argus Core
        self.output , self.tokens_used = self.search_engine.run_full_research(input_prompt=self.text_input.value)
        
        # Create a new Output object to display the chatbot response
        if self.first_cycle:
            result = Output(output_text=self.output, input_text=self.text_input.value, msg_index=1, output_delete=self.outputDelete, first_cycle=self.first_cycle)
            self.first_cycle = False
        else:
            # TODO
            pass
        
        # Add the Output object to the output column
        self.output_column.controls.append(result)
        
        # Clear the text input field
        self.text_input.value = ""
        
        # Update the page to reflect the changes
        self.update()

    def outputDelete(self, result):
        # Remove the specified result from the output column
        self.output_column.controls.remove(result)
        
        # Update the page to reflect the changes
        self.update()


# Define a class called Output that is a subclass of ft.UserControl
class Output(ft.UserControl):
    def __init__(self, output_text:str, input_text:str, msg_index:int, output_delete, first_cycle:bool):
        super().__init__()
        self.output = output_text 
        self.input = input_text
        self.myoutput_delete = output_delete
        self.first_cycle = first_cycle
        self.msg_index = msg_index

    def build(self):

        # Assemble markdown output
        
        self.md1 = f"### > User:\n{self.input}\n### > Argus:\n{self.output}"

        if self.first_cycle:
            # If this is the first message -> add sources
            self.md1 += f"\n### Sources:\nToDo"
        
        
        # Create a delete button to remove the chatbot response
        if self.first_cycle:
            self.delete_button = ft.IconButton(ft.icons.DELETE_OUTLINE_SHARP, on_click=self.delete)
        

        self.md_display = ft.Markdown(
                                        self.md1,
                                        selectable=True,
                                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                        on_tap_link=lambda e: print(e.data),
                                    )
        
        # Create a column layout to arrange the elements
        self.display_view = ft.Column(controls=[ self.md_display, self.delete_button])

        # Return the column layout as the UI representation of the Output class
        return self.display_view

    def delete(self, e):
        # Call the outputDelete function with the current instance as an argument
        self.myoutput_delete(self)


# Define a main function that sets up the page layout
def main(page: ft.page):
    page.scroll = True
    page.window_width = 900
    page.window_height = 700
    
    # Create a new Chat object
    mychat = Chat()
    
    # Add the Chat object to the page
    page.add(mychat)


# Run the application using the ft.app() function and pass the main function as the target
ft.app(target=main)