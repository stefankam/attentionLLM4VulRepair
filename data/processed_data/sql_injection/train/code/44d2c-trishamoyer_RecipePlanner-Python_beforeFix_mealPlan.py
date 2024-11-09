from tkinter import *
import datetime
from isoweek import Week
from tkinter import ttk
import sqlite3
from PIL import Image, ImageTk

LARGE_FONT=("Trebuchet MS", 24)
MEDIUM_FONT=("Trebuchet MS", 14)


class MakeMealPlan(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg="#f8f8f8")
        menuFrame = Frame(self, bg="#e7e7e7")
        menuFrame.pack(fill='both')
        load = Image.open("home.jpg")
        render = ImageTk.PhotoImage(load)
        from landingpage import LandingPage
        img = Button(menuFrame, image=render, borderwidth=0, highlightthickness=0, highlightbackground="#e7e7e7",
                     command=lambda: controller.show_frame(LandingPage))
        img.image = render
        img.pack(side=LEFT)

        label = Label(menuFrame, text="Meal Planner", font=LARGE_FONT, bg="#e7e7e7", fg="#272822")
        label.pack(side=LEFT, padx=289)

        groceryButton = Button(menuFrame, text="Grocery List", highlightbackground="#e7e7e7", command=lambda: view_grocery_list())
        groceryButton.pack(side=LEFT)

        viewRecipeFrame = Frame(self, bg="#f8f8f8")

        now = datetime.datetime.now()
        dt = datetime.date(now.year, now.month, now.day)
        weekNumber = dt.isocalendar()[1]
        w = Week(now.year, weekNumber)

        menu = Frame(self, bg="#f8f8f8")
        menu.rowconfigure(0, weight=1)
        menu.columnconfigure(0, weight=1)
        menu.rowconfigure(1, weight=3)
        menu.columnconfigure(1, weight=3)
        menu.pack()

        columnLabels = ["Breakfast", "Lunch", "Dinner"]
        for i in range(len(columnLabels)):
            Label(menu, text=columnLabels[i], font=("Trebuchet MS", 16), bg="#f8f8f8").grid(row=0, column=i+2, pady= 10,
                                                                                        padx=85, sticky="nsew")
        mondayText = "Monday " + str(w.monday())
        tuesdayText = "Tuesday " + str(w.tuesday())
        wednesdayText = "Wednesday " + str(w.wednesday())
        thursdayText = "Thursday " + str(w.thursday())
        fridayText = "Friday " + str(w.friday())
        saturdayText = "Saturday " + str(w.saturday())
        sundayText = "Sunday " + str(w.sunday())

        labels = [mondayText, tuesdayText, wednesdayText, thursdayText, fridayText, saturdayText, sundayText]
        for i in range(len(labels)):
            Label(menu, font=("Trebuchet MS", 12), bg="#f8f8f8", text=labels[i]).grid(row=i+1, column=0, padx = 5, pady=15, sticky="w")
            sep = ttk.Separator(menu, orient="vertical")
            sep.grid(row=i+1, column=1, padx=5, sticky="nsew")

        buttonDict = {}
        listOfButtons = []
        for rows in range(len(labels)):
            for columns in range(len(columnLabels)):
                buttons = Button(menu, text="Add meal to day", highlightbackground="#f8f8f8", command=lambda x=rows + 1, y=columns + 2: add_meal(x, y))
                buttons.grid(row=rows+1, column=columns+2)
                buttons.position = (rows+1, columns+2)
                buttonDict[buttons] = buttons.position
                listOfButtons.append(buttons)

        def add_meal(rowLocation, columnLocation):
            menu.pack_forget()
            viewRecipeFrame.forget()
            add_meal_frame = Frame(self, bg="#f8f8f8")
            add_meal_frame.rowconfigure(0, weight=1)
            add_meal_frame.columnconfigure(0, weight=1)
            add_meal_frame.rowconfigure(1, weight=3)
            add_meal_frame.columnconfigure(1, weight=3)
            add_meal_frame.pack()

            recipeNames = []
            ingredientList = []
            database_file = "meal_planner.db"
            with sqlite3.connect(database_file) as conn:
                cursor = conn.cursor()
                selection = cursor.execute("""SELECT * FROM recipe""")
                for result in [selection]:
                    for row in result.fetchall():
                        name = row[0]
                        ingredients = row[4]
                        recipeNames.append(name)
                        ingredientList.append(ingredients)
            for i in range(len(recipeNames)):
                Button(add_meal_frame, text=recipeNames[i], highlightbackground="#f8f8f8", command=lambda x=recipeNames[i], y=ingredientList[i]:add_recipe(x, y, add_meal_frame,
                                                                                     rowLocation, columnLocation)).grid(row=i, column=0)

        def add_recipe(recipe, ingredients, view, row, column):
            view.pack_forget()
            viewRecipeFrame.forget()
            searchIndex = (row, column)
            for key, value in buttonDict.items():
                if value == searchIndex:
                    key.destroy()
            save_weeks_recipes(recipe, row, column)
            save_ingredients(ingredients)
            recipeLabel = Label(menu, text=recipe, bg="#f8f8f8")
            recipeLabel.grid(row = row, column = column)
            recipeLabel.bind("<Button-1>", lambda event: callback(recipe))
            menu.pack()

        def callback(recipeName):
            menu.pack_forget()
            viewRecipeFrame.pack(expand=True, fill='both')
            groceryButton.pack_forget()
            database_file = "meal_planner.db"
            print(recipeName)
            with sqlite3.connect(database_file) as conn:
                cursor = conn.cursor()
                <vul/>selection = cursor.execute("""SELECT * FROM recipe WHERE name = """ + "\"" + recipeName + "\"")</vul>
                for result in [selection]:
                    for row in result.fetchall():
                        name = row[0]
                        time = row[1]
                        servings = row[2]
                        ingredients = row[4]
                        directions = row[5]

                        string = ("Name: {} \n Cook time: {} \n Number of Servings: {} \n ".format(name, time, servings))
                        secondString = ("Ingredients: {}".format(ingredients))
                        thirdString = ("Directions: {}".format(directions))
            Label(viewRecipeFrame, text=string, font=MEDIUM_FONT, bg="#f8f8f8", fg="#000000").pack(side=TOP)
            Label(viewRecipeFrame, text=secondString, font=MEDIUM_FONT, bg="#f8f8f8", fg="#000000").pack(side=TOP)
            Label(viewRecipeFrame, text=thirdString, font=MEDIUM_FONT, bg="#f8f8f8", fg="#000000").pack(side=TOP)
            returnButton = Button(menuFrame, text = "Return to Menu", highlightbackground="#e7e7e7", command=lambda: [viewRecipeFrame.pack_forget(),
                                                                                     menu.pack(), returnButton.pack_forget(), label.configure(text="Meal Planer"),
                                                                                    groceryButton.pack(side=RIGHT)])
            returnButton.pack(side=RIGHT)


        def view_grocery_list():
            print("grocery== list")
            groceryListFrame = Frame(self)
            groceryListFrame.rowconfigure(0, weight=1)
            groceryListFrame.columnconfigure(0, weight=1)
            groceryListFrame.rowconfigure(1, weight=3)
            groceryListFrame.columnconfigure(1, weight=3)
            groceryListFrame.pack()

            menu.pack_forget()
            groceryButton.pack_forget()
            label.configure(text="Grocery List")

            i = 0
            database_file = "meal_planner.db"
            item_array = []
            with sqlite3.connect(database_file) as conn:
                cursor = conn.cursor()
                tableName = "ingredients_" + str(weekNumber)
                <vul/>selection = cursor.execute("""SELECT * FROM """ + tableName)</vul>
                for result in [selection]:
                    for row in result.fetchall():
                        print(row)
                        for ingredient in row:
                            print(ingredient)
                            item_array.append(str(ingredient).split())
                        i = i +1
                        Label(groceryListFrame, text=ingredient, font=MEDIUM_FONT, justify=LEFT).grid(row=i, column=0, sticky="w")
            

            j = 0
            for item in item_array:
                print(item)


            returnButton = Button(menuFrame, text = "Return to Menu", highlightbackground="#e7e7e7", command=lambda: [groceryListFrame.pack_forget(),
                                                                                     menu.pack(), returnButton.pack_forget(), label.configure(text="Meal Planer"),
                                                                                    groceryButton.pack(side=RIGHT)])
            returnButton.pack(side=RIGHT)


        def save_ingredients(ingredients):
            database_file = "meal_planner.db"
            with sqlite3.connect(database_file) as conn:
                # create the table if it hasn't been created yet
                tableName = "ingredients_" + str(weekNumber)
                conn.execute('''CREATE TABLE IF NOT EXISTS ''' + tableName + ''' (ingredients text)''')
                conn.execute("""INSERT INTO """ + tableName + """ VALUES (?);""", (ingredients,))
            
        def save_weeks_recipes(recipeName, row, column):
            print("save weeks")
            database_file = "meal_planner.db"
            with sqlite3.connect(database_file) as conn:
                # create the table if it hasn't been created yet
                tableName = "recipes_" + str(weekNumber)
                conn.execute('''CREATE TABLE IF NOT EXISTS ''' + tableName + ''' (recipe text, row int, column int)''')
                conn.execute("""INSERT INTO """ + tableName + """ VALUES (?, ?, ?);""", (recipeName, row, column))
            