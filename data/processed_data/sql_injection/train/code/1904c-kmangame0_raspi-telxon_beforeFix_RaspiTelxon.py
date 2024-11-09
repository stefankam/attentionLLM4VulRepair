<vul/></vul>import sys
import tkinter as tk
import Controller as dbc
from PIL import Image, ImageTk
from tkinter import font


class RaspiTelxon(tk.Tk):
	
	def __init__(self):
		
		#initialize root object
		tk.Tk.__init__(self)

		self.title("Raspi-Telxon")
		
		self.titleFont = font.Font(family='Helvetica', size=24)
		self.itemFont = font.Font(family='Helvetica', size=18)

		# Self is this instance of Tk IE:- "root"
		container = tk.Frame(self)

		container.pack(side="top", fill="both", expand=True)
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		# dispatch_dict = {"SearchPage" : SearchPage}
		# self.dispatch_dict = dispatch_dict
		
		self.frames = {}
		self.result = ""
		self.container = container

		for F in (StartPage, SearchPage):
			
			frame = F(container, self)

			#print(F.__name__)

			self.frames[F] = frame

			#print(self.frames)

			frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(StartPage)


	def create_frame(self, F):

		new_frame = SearchPage(self.container, self)

		self.frames[SearchPage] = new_frame

		new_frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(new_frame)
		

	def remove_frame(self, frame):
		
		print("remove_frame: " + str(frame))

		self.frames.pop(frame, None)

	def show_frame(self, cont):

		frame = self.frames[cont]
		
		frame.tkraise()

	# Just break out the for on line 23 into a function
	# Reduce duplicate code
	def custom_frame(self):
		
		result_frame = ResultsPage(self.container, self)
		
		self.frames[ResultsPage] = result_frame
		
		result_frame.grid(row=0, column=0, sticky="nsew")
		
		self.show_frame(ResultsPage)

	def set_result(self, result):
		self.result = result

	def get_result(self):
		return self.result


class StartPage(tk.Frame):

	def __init__(self, parent, controller):

		tk.Frame.__init__(self, parent)

		label = tk.Label(self, text = "Login Page", font=controller.titleFont)
		label.pack(pady=10, padx=10)

		enterAppButton = tk.Button(self, text="Start Using Raspi-Telxon!", 
			font=controller.itemFont,command=lambda: controller.show_frame(SearchPage))

		enterAppButton.pack(pady=5)

		exitAppButton = tk.Button(self, text="Quit", 
			font=controller.itemFont, command=lambda: sys.exit(0))

		exitAppButton.pack(pady=5)

class SearchPage(tk.Frame):

	def __init__(self, parent, controller):

		tk.Frame.__init__(self, parent)

		self.controller = controller

		statusbar = tk.Frame(self)
		statusbar.pack(side="top", fill="x")
		self.statusbar = statusbar

		navbar = tk.Frame(self)
		navbar.pack(side="bottom", fill="x")
		self.navbar = navbar

		UPC_Label = tk.Label(self, text="UPC", font=controller.titleFont)
		UPC_Label.pack(pady=10, padx=10, anchor="center")

		self.UPC_Entry = tk.Entry(self)
		self.UPC_Entry.pack(pady=10, padx=10, anchor="center")

		backButton = tk.Button(navbar, text="Back",
			font=controller.itemFont, command=lambda: controller.show_frame(StartPage))
		backButton.pack(side="left", pady=10, padx=10)

		Search_Button = tk.Button(navbar, text="Search", 
															font=controller.itemFont, command=self.search)
		Search_Button.pack(side="left", pady=10, padx=10)

		exitAppButton = tk.Button(navbar, text="Quit", 
			font=controller.itemFont, command=lambda: sys.exit(0))
		exitAppButton.pack(side="left", pady=10, padx=10)

		<vul/># temp = self.winfo_children()
		# print(temp)</vul>


	<vul/>def search(self):
		
		upc = ""

		upcEntry = self.UPC_Entry.get()</vul>

		if(upcEntry == ""):
			emptyInputLabel = tk.Label(self.statusbar, text="UPC Cannot Be Empty", fg="red")
			emptyInputLabel.pack()

		if(self.UPC_Entry.get() != ""):

			<vul/>self.View_Result_Button = tk.Button(self.navbar, text="View Result", 
				font=self.controller.itemFont, command=lambda: self.controller.custom_frame())</vul>

			<vul/>self.View_Result_Button.pack(side="left", pady=10, padx=10)</vul>

			<vul/>upc = self.UPC_Entry.get()</vul>

			database = dbc.DB_Connector()

			database.some_upc = upc

			result = database.fetch_product()
		
			self.controller.set_result(result)

			if(result is None):
				<vul/>result_not_found = tk.Label(self, text="No Result Found!", font=self.controller.itemFont)
				result_not_found.pack()
				self.View_Result_Button.config(state='disabled')</vul>
			elif(result is not None):
				<vul/>result_found_notification = tk.Label(self, text="Results Found!", font=self.controller.itemFont)
				result_found_notification.pack()
				self.View_Result_Button.config(state='normal')</vul>


class ResultsPage(tk.Frame):

	def __init__(self, parent, controller):

		tk.Frame.__init__(self, parent)

		self.controller = controller

		<vul/>(ID, UPC, name, imageURI) = controller.get_result()</vul>

		<vul/>load = Image.open(imageURI)
		render = ImageTk.PhotoImage(load)
		
		img_label = tk.Label(self, image=render)
		img_label.image = render
		img_label.pack(side="right")</vul>

		<vul/>name_label = tk.Label(self, text="Product: " + name, font=controller.titleFont)
		name_label.pack(pady=10, padx=10, anchor="nw")</vul>

		<vul/>upc_label = tk.Label(self, text="UPC: " + UPC, font=controller.itemFont)
		upc_label.pack(pady=10, padx=10, anchor="nw")</vul>

		<vul/>new_search_button = tk.Button(self, text="New Search",
			font=controller.itemFont, command=lambda: self.new_search())</vul>

		<vul/>new_search_button.pack(side="left", pady=10, padx=10, anchor="sw")</vul>

		<vul/>exit_app_button = tk.Button(self, text="Quit",</vul> 
			font=controller.itemFont, command=lambda: sys.exit(0))

		<vul/>exit_app_button.pack(side="left", pady=10, padx=10, anchor="sw")</vul>


	<vul/>def new_search(self):</vul>

		<vul/>self.controller.remove_frame(SearchPage)</vul>

		<vul/>new_frame = self.controller.create_frame(SearchPage)</vul>

		<vul/>#print("IN NEW SEARCH: " + str(type(new_frame)))</vul>

		#self.controller.show_frame(new_frame)

		# self.controller.frames.pop('SearchPage', None)

		<vul/># frame = SearchPage(self.controller.container, self.controller)</vul>

		<vul/># self.controller.frames[SearchPage] = frame</vul>

		<vul/># self.controller.show_frame(SearchPage)</vul>



app = RaspiTelxon()
app.geometry("800x480")
app.mainloop()