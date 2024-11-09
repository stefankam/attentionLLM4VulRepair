<fix/></fix>import sys
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
		
		self.frames = {}
		self.result = ""
		self.container = container

		for F in (StartPage, SearchPage):
			
			frame = F(container, self)

			self.frames[F] = frame

			frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(StartPage)


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

		<fix/>statusbar = tk.Frame(self)
		statusbar.pack(side="top", fill="x")
		self.statusbar = statusbar</fix>

		<fix/>navbar = tk.Frame(self)
		navbar.pack(side="bottom", fill="x")
		self.navbar = navbar</fix>

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

		<fix/>viewResultButton = tk.Button(navbar, text="View Result", 
				font=controller.itemFont, command=lambda: controller.custom_frame())
		self.viewResultButton = viewResultButton
		viewResultButton.pack(side="left", pady=10, padx=10)
		viewResultButton.config(state='disabled')</fix>

		self.emptyInputLabelVar = tk.StringVar()
		emptyInputLabelVar = self.emptyInputLabelVar
		tk.Label(statusbar, textvariable = emptyInputLabelVar, fg="red").pack()

		<fix/>self.resultNotificationVar = tk.StringVar()
		resultNotificationVar = self.resultNotificationVar
		tk.Label(self, textvariable = resultNotificationVar, font=controller.itemFont).pack()</fix>


	<fix/>def search(self):</fix>

		<fix/>upc = self.UPC_Entry.get()</fix>

		<fix/>if(upc == ""):
			self.emptyInputLabelVar.set("UPC Must Not Be Empty")</fix>

		<fix/>if(upc != ""):</fix>

			database = dbc.DB_Connector()

			database.some_upc = upc

			result = database.fetch_product()

			self.controller.set_result(result)

			if(result is None):
				<fix/>self.resultNotificationVar.set("No Results Found!")</fix>
			elif(result is not None):
				<fix/>self.resultNotificationVar.set("Results Found!")
				self.viewResultButton.config(state='normal')</fix>


class ResultsPage(tk.Frame):

	def __init__(self, parent, controller):

		tk.Frame.__init__(self, parent)

		self.controller = controller

		<fix/>(ID, UPC, name, imageURI, unitsInPackage, unitSize, ingredients) = controller.get_result()</fix>

		statusbar = tk.Frame(self)
		statusbar.pack(side="top", fill="x")
		self.statusbar = statusbar

		navbar = tk.Frame(self)
		navbar.pack(side="bottom", fill="x")
		self.navbar = navbar

		<fix/>imageAnchor = tk.Frame(self)
		imageAnchor.pack(side="right", fill="y")
		self.imageAnchor = imageAnchor</fix>

		<fix/>main = tk.Frame(self)
		main.pack(side="left", fill="both")
		self.main = main</fix>

		<fix/>new_search_button = tk.Button(navbar, text="New Search",
			font=controller.itemFont, command=lambda: self.new_search())
		new_search_button.pack(side="left", pady=10, padx=10)</fix>

		<fix/>exit_app_button = tk.Button(navbar, text="Quit",</fix> 
			font=controller.itemFont, command=lambda: sys.exit(0))
		exit_app_button.pack(side="left", pady=10, padx=10)

		<fix/>load = Image.open(imageURI)
		render = ImageTk.PhotoImage(load)
		img_label = tk.Label(imageAnchor, image=render)
		img_label.image = render
		img_label.pack()</fix>

		name_label = tk.Label(main, wraplength=300, text=name, font=controller.itemFont)
		name_label.pack(pady=10, padx=10, anchor="ne")

		<fix/>upc_label = tk.Label(main, text="UPC:- " + UPC, font=controller.itemFont)
		upc_label.pack(pady=5, padx=5, anchor="nw")</fix>

		<fix/>unit_size_label = tk.Label(main, text="Unit Size:- " + unitSize, font=controller.itemFont)
		unit_size_label.pack(pady=5, padx=5, anchor="nw")</fix>

		<fix/>units_in_package_label = tk.Label(main, text="Units In Package:- " + unitsInPackage, font=controller.itemFont)
		units_in_package_label.pack(pady=5, padx=5, anchor="nw")</fix>

		<fix/>ingredients_label = tk.Label(main, wraplength=350, text="Ingredients\n" + ingredients, font=controller.itemFont)
		ingredients_label.pack(pady=5, padx=5, anchor="n")</fix>


	<fix/>def new_search(self):

		self.controller.frames.pop(SearchPage, None)</fix>

		<fix/>newSearchFrame = SearchPage(self.controller.container, self.controller)</fix>

		<fix/>self.controller.frames[SearchPage] = newSearchFrame</fix>

		<fix/>newSearchFrame.grid(row=0, column=0, sticky="nsew")</fix>

		self.controller.show_frame(SearchPage)


app = RaspiTelxon()
app.geometry("800x480")
app.mainloop()