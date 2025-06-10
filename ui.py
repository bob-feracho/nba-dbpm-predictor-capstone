import tkinter as tk
from data import Models

class App:
    def __init__(self, root):
        #Set up Inputs and Labels
        self.root = root
        self.root.title("NBA Position based DBPM predictor")
        self.root.geometry("600x400")
        tk.Label(self.root, text="height_wo_shoes").grid(row=0, column=0)
        self.height_wo_shoes_entry = tk.Entry(self.root)
        self.height_wo_shoes_entry.insert(0, "90")
        self.height_wo_shoes_entry.grid(row=0, column=1)
        tk.Label(self.root, text="weight").grid(row=1, column=0)
        self.weight = tk.Entry(self.root)
        self.weight.insert(0, "190")
        self.weight.grid(row=1, column=1)
        tk.Label(self.root, text="wingspan").grid(row=2, column=0)
        self.wingspan = tk.Entry(self.root)
        self.wingspan.insert(0, "90")
        self.wingspan.grid(row=2, column=1)
        tk.Label(self.root, text="standing_vertical_leap").grid(row=3, column=0)
        self.standing_vertical_leap = tk.Entry(self.root)
        self.standing_vertical_leap.insert(0, "35")
        self.standing_vertical_leap.grid(row=3, column=1)
        tk.Label(self.root, text="lane_agility_time").grid(row=4, column=0)
        self.lane_agility_time = tk.Entry(self.root)
        self.lane_agility_time.insert(0, "3.5")
        self.lane_agility_time.grid(row=4, column=1)
        tk.Label(self.root, text="three_quarter_sprint").grid(row=5, column=0)
        self.three_quarter_sprint = tk.Entry(self.root)
        self.three_quarter_sprint.insert(0, "3.5")
        self.three_quarter_sprint.grid(row=5, column=1)
        tk.Label(self.root, text="max_vertical_leap").grid(row=6, column=0)
        self.max_vertical_leap = tk.Entry(self.root)
        self.max_vertical_leap.insert(0, "42")
        self.max_vertical_leap.grid(row=6, column=1)
        tk.Label(self.root, text="bench_press").grid(row=7, column=0)
        self.bench_press = tk.Entry(self.root)
        self.bench_press.insert(0, "0")
        self.bench_press.grid(row=7, column=1)
        tk.Button(self.root, text="Submit", command=self.assignAndPredict).grid(row=9, column=1)
        tk.Label(self.root, text="pos (pg, sg, sf, pf, c)").grid(row=8, column=0)
        self.pos= tk.Entry(self.root)
        self.pos.insert(0, "pg")
        self.pos.grid(row=8, column=1)
        self.playerPosition = 'pg'


#       Buttons for Metrics
        tk.Button(self.root, text="R^2 For Position", command=self.getRSquared).grid(row=1, column=2)

        #Buttons for graphs
        tk.Button(self.root, text="Feature Importance Graph For Position", command=self.generateFeatureImportanceGraphsBasedOnPos).grid(row=0, column=3)
        tk.Button(self.root, text="SHAP Graphs For Position", command=self.generateShapGraphsBasedOnPos).grid(row=1, column=3)
        tk.Button(self.root, text="Generate Scatterplots Graph For Position", command=self.generateScatterplotsBasedOnPos).grid(row=2, column=3)

        self.the_models= Models()
        self.root.mainloop()

    def assignData(self):
        try:
            text_boxes = [
                self.height_wo_shoes_entry,
                self.weight,
                self.wingspan,
                self.three_quarter_sprint,
                self.lane_agility_time,
                self.standing_vertical_leap,
                self.max_vertical_leap,
                self.bench_press
            ]
            ind = 0
            results = []
            for text_box in text_boxes:
                val = text_box.get()
                if not self.is_valid_float(val) and ind != 7:
                    print("invalid data. Data must be float.")
                    break
                if not self.is_valid_int(val) and ind == 7:
                    print("invalid. Bench press must be an integer")
                    break
                if ind != 7:
                    val = float(val)
                else:
                    val = int(val)
                if not self.is_valid_within_guards(val, ind):
                    print(""" Data must be within Guards
                    height_wo_shoes: 60–90 inches (5'0" to 7'6")
                            weight: 120–350 lbs
                            wingspan: 60–100 inches
                            sprint, agility	2.5–5.0 sec
                            vertical_leap standing, max 20–50 inches
                            bench_press 0–30 reps""")
                    break
                results.append(val)
                # print(f"val: {val}")
                # print(f"ind: {ind}")
                ind += 1



            self.playerPosition = self.pos.get()
            return results
        except ValueError as e:
            print("🚫 Invalid input — make sure all fields are filled with numeric values.")
            return None

    def assignAndPredict(self):
        temp = tk.Tk()
        results = self.assignData()
        # print("assignData() output:", results)
        predicted_dbpm = self.the_models.makePrediction(self.playerPosition, results)
        temp.geometry("200x200")
        tk.Label(temp, text=f"Predicted DBPM: {predicted_dbpm}").pack()


    def getRSquared(self):
        self.playerPosition = self.pos.get()
        temp = tk.Tk()
        (tk.Label(temp, text=f"Training R^2 Score: {self.the_models.models[self.playerPosition]["train-test-r2_mse"][0]}"
                            f"\nTest R^2 Score: {self.the_models.models[self.playerPosition]["train-test-r2_mse"][1]}"
                            f"\nMean Squared Error: {self.the_models.models[self.playerPosition]["train-test-r2_mse"][2]}")
         .pack())


    def generateFeatureImportanceGraphsBasedOnPos(self):
        self.playerPosition = self.pos.get()
        self.the_models.generateFeatureImportanceGraph(self.playerPosition)

    def generateScatterplotsBasedOnPos(self):
        self.playerPosition = self.pos.get()
        self.the_models.generateScatterplotsOnTestandTrainSet(self.playerPosition)

    def generateShapGraphsBasedOnPos(self):
        self.playerPosition = self.pos.get()
        temp = tk.Tk()
        temp.geometry("300x300")
        tk.Label(temp, text="bar graph or beeswarm?").grid(row=0, column=1)
        tk.Button(temp, text="bar", command=self.shapGenerationForBar).grid(row=1, column=0)
        tk.Button(temp, text="beeswarm", command=self.shapGenerationForBeehive).grid(row=1, column=2)

    def shapGenerationForBar(self):
        self.playerPosition = self.pos.get()
        self.the_models.generateShapData(self.playerPosition, 0)
    def shapGenerationForBeehive(self):
        self.playerPosition = self.pos.get()
        self.the_models.generateShapData(self.playerPosition, 1)

    def is_valid_float(self, val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    def is_valid_int(self, val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    def is_valid_within_guards(self, val, ind):
        if ind == 0 and 60 <= val <= 90:  # height
            return True
        elif ind == 1 and 120 <= val <= 350:  # weight
            return True
        elif ind == 2 and 60 <= val <= 100:  # wingspan
            return True
        elif ind in [3, 4] and 2.0 <= val <= 6.0:  # agility/sprint
            return True
        elif ind in [5, 6] and 20 <= val <= 50:  # verticals
            return True
        elif ind == 7 and 0 <= val <= 30:  # bench press
            return True
        return False