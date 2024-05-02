import flet as ft

from argus_src.core import ArgusWebsearch

from pandas import DataFrame

# https://github.com/flet-dev/flet/discussions/2749


# Define a class called Chat that is a subclass of ft.UserControl

# Define a class called Output that is a subclass of ft.UserControl
class Output(ft.UserControl):
    def __init__(self, output_text:str, input_text:str, sequence_num:int, output_delete, output_relate, first_cycle:bool, sources:DataFrame=None):
        super().__init__()
        self.output = output_text 
        self.input = input_text
        self.sources = sources
        self.myoutput_delete = output_delete
        self.myoutput_relate = output_relate
        self.first_cycle = first_cycle
        self.sequence_num = sequence_num

    def build(self):

        # Assemble markdown output
        
        #self.md1 = f"# > User:\n{self.input}\n### > Argus:\n{self.output}"


        self.md1 = f"{self.output}\n\n"

        if self.first_cycle:
            if len(self.sources) > 0:
                # If sources list was added and is larger than 0 -> add sources
                self.md1 += f"\n\n | `Source no.` | `URL` |\n| --- | ------------ |\n"

                # Print unique source URLs
                urls_added = []
                url_numbers = []

                src_num = 1
                for index, doc in self.sources.iterrows():

                    # Add URL only once
                    if doc['URL'] in urls_added:
                        # Source is alread added
                        # Get index num
                        src_idx = urls_added.index(doc['URL'])

                        # Add source number to array
                        url_numbers[src_idx].append(str(src_num))

                    else:
                        # Source is not present in array
                        urls_added.append(doc['URL'])
                        url_numbers.append([str(src_num)])
                    
                    src_num += 1

                # Print URLs and source numbers to output
                i = 0
                for url_src in urls_added:
                    src_numbers = ', '.join(url_numbers[i])

                    # print(f"{src_numbers}: {url_src}")

                    self.md1 += f"|{src_numbers}|{url_src}|\n"

                    i += 1
        
        
        # Create a delete button to remove the chatbot response
        # if self.msg_index != 1:
        self.delete_button = ft.IconButton(ft.icons.DELETE_OUTLINE_SHARP, on_click=self.delete)

        self.relate_button = ft.IconButton(ft.icons.REPLY_ROUNDED, on_click=self.relate)
        

        self.md_display = ft.Markdown(
                                        self.md1,
                                        selectable=True,

                                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                        on_tap_link=lambda e: print(e.data),
                                    )

        self.circle_avatar_argus = ft.CircleAvatar(
                                                    content=ft.Text("A"),
                                                    color=ft.colors.WHITE,
                                                    bgcolor=ft.colors.LIGHT_BLUE_ACCENT_700
                                                    )

        self.circle_avatar_user = ft.CircleAvatar(
                                                    content=ft.Text("U"),
                                                    color=ft.colors.WHITE,
                                                    bgcolor=ft.colors.TEAL_ACCENT_700
                                                    )
        
        self.controls_row_user = ft.Container( content= ft.Row([
                                                            self.circle_avatar_user,
                                                            ft.Column(
                                                                        [
                                                                            ft.Text("User", weight="bold"),
                                                                            ft.Text(self.input, selectable=True)
                                                                        ],
                                                                        tight=True,
                                                                        spacing=5,
                                                                    ),
                                                            ]),
                                                            padding=10,
                                                            border_radius=5,
                                                            # bgcolor=ft.colors.BLUE_GREY_100,
                                                            border=ft.border.all(1, ft.colors.GREY))
        
        self.controls_row_argus = ft.Container( content= ft.Row([
                                                                    ft.Row([
                                                                        self.circle_avatar_argus,
                                                                        ft.Column(
                                                                                    [
                                                                                        
                                                                                        ft.Text("Argus", weight="bold"),
                                                                                        
                                                                                    ],
                                                                                    tight=False,
                                                                                    spacing=5,
                                                                                ),

                                                                        ],

                                                                    ),
                                                                    self.md_display
                                                            ],

                                                            wrap=True),
                                                            padding=10,
                                                            border_radius=5,
                                                            # bgcolor=ft.colors.BLUE_GREY_100,
                                                            border=ft.border.all(1, ft.colors.GREY))
        
        # Add delete button only after 1st cycle
        if self.first_cycle:
            delete_relate_buttons = ft.Row([self.relate_button])
        else:
            delete_relate_buttons = ft.Row([self.relate_button, self.delete_button])
        
        # Create a column layout to arrange the elements
        self.display_view = ft.Column(controls=[ self.controls_row_user, self.controls_row_argus, delete_relate_buttons])

        # Return the column layout as the UI representation of the Output class
        return self.display_view

    def delete(self, e):
        # Call the outputDelete function with the current instance as an argument
        self.myoutput_delete(self)

    def relate(self, e):
        # Call the outputRelate function with the current instance as an argument
        self.myoutput_relate(self)

    def get_index(self)->int:
        return self.sequence_num


# Define a main function that sets up the page layout
def main(page: ft.page):

    page.theme_mode = ft.ThemeMode.LIGHT
    
    global first_cycle
    first_cycle = True

    global sequence_num
    sequence_num = 0

    # Init Argus core
    global search_engine
    search_engine = ArgusWebsearch()

    def outputDelete(result):
        # Remove the specified result from the conversation store
        print(f"--> Going to delete conversation sequence {result.get_index()}...")
        search_engine.conversation_stage4.delete_sequence(result.get_index())

        # Remove the specified result from the output column
        output_column.controls.remove(result)
        
        # Update the page to reflect the changes
        page.update()

        # print(search_engine.conversation_stage4.output_msg_store_cli())

    def outputRefer(result):
        # Get sequence num of output element and write it to the text input field
        text_input.value = f">> Refer to {result.get_index()}: {text_input.value}"
        page.update()
        
    def btn_start_new_clicked(e):
        global search_engine
        global first_cycle
        global sequence_num

        first_cycle = True
        sequence_num = 0

        # Delete all UI elements
        output_column.controls.clear()
        page.update()

        # Reset all storages
        search_engine.clear()

    def btn_send_clicked(e):

        global first_cycle
        global sequence_num
 
        if text_input.value == "": return

        # Activate wait animation
        wait_animation = ft.Column(
                            [ft.ProgressRing(), ft.Text("Diving in the deep space of the internet...")],
                        )
        output_column.controls.append(wait_animation)
        page.update()


        # Send the user input to the Argus Core
        output = search_engine.run_task(input_prompt=text_input.value)
  
        # Test output
        '''output = f"HeyHey \n\n| Source no. | URL | \n| --- | --- | \n| 123 | Test |"
        tokens_used = 123
        sources = []
        sources.append(["Test", "123"])
        sources = DataFrame(sources, columns =['URL', 'Content'])'''

        '''output = f"This is my answer..."
        tokens_used = 123
        sequence_num += 1'''

        if output[0] == "load":
            # Load existing conversation
            print("UI loading...")

            # Removing wait animation
            output_column.controls.remove(wait_animation)
            # Clear the text input field
            text_input.value = ""
            page.update()

        else:
            # Continue conversation

            if first_cycle:
                # Get context used for answer
                sources = search_engine.rag_context
            else:
                # Init sources since it is not used
                sources = []

            sequence_num = search_engine.conversation_stage4.sequence_num - 1

            # Create a new Output object to display the chatbot response
            result = Output(
                                output_text = output[1][0],
                                input_text = text_input.value,
                                sequence_num = sequence_num,
                                output_delete = outputDelete,
                                output_relate = outputRefer,
                                first_cycle=first_cycle,
                                sources=sources
                            )
            
            first_cycle = False
            
            # Removing wait animation
            output_column.controls.remove(wait_animation)
            page.update()

            # Add the Output object to the output column
            output_column.controls.append(result)
            
            # Clear the text input field
            text_input.value = ""
            
            # Update the page to reflect the changes
            page.update()


    # Create a heading text element
    heading = ft.Text(value="Argus AI Web Research", size=24)

    # Create a text input field for user prompts
    text_input = ft.TextField(hint_text="Enter your prompt", expand=True, multiline=False, on_submit=btn_send_clicked)

    # Send button
    send_btn = ft.IconButton(icon=ft.icons.SEND_ROUNDED, on_click=btn_send_clicked)

    # Restart button
    new_btn = ft.IconButton(icon=ft.icons.RESTART_ALT_ROUNDED, on_click=btn_start_new_clicked)

    # Create an empty column to hold the output elements
    output_column = ft.ListView(expand=True,
                                    spacing=10,
                                    auto_scroll=True,   )

    page.add(                   heading,
                                ft.Container(
                                    content=output_column,
                                    #border=ft.border.all(0, ft.colors.OUTLINE),
                                    border_radius=5,
                                    padding=10,
                                    expand=True,
                                    ),

                                ft.Row(
                                        controls=[
                                                    text_input,
                                                    send_btn,
                                                    new_btn
                                                ],
                                    ))
    
    


# Run the application using the ft.app() function and pass the main function as the target
ft.app(target=main)
