from xxsubtype import bench

import kaggle
import sqlite3
import pandas
import shap
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as pyplot
import os


class Models:

    def __init__(self):
        self.downloadData()
        self.query =  """
            SELECT 
                `draft_combine_stats`.`player_name`,
                `draft_combine_stats`.`height_wo_shoes`,
                `draft_combine_stats`.`weight`,
                `draft_combine_stats`.`wingspan`,
                `draft_combine_stats`.`standing_vertical_leap`,
                `draft_combine_stats`.`max_vertical_leap`,
                `draft_combine_stats`.`lane_agility_time`,
                `draft_combine_stats`.`three_quarter_sprint`, 
                `draft_combine_stats`.`bench_press`, 
                AVG(`advancedStats_db`.`advanced-stats`.`dbpm`) as avg_dbpm 
            FROM `draft_combine_stats`
            LEFT JOIN advancedStats_db.`advanced-stats`
                ON `draft_combine_stats`.`player_name` = `advancedStats_db`.`advanced-stats`.`player`
            WHERE `draft_combine_stats`.`three_quarter_sprint` IS NOT NULL
                AND `draft_combine_stats`.`season` >= 2000
                AND `draft_combine_stats`.`season` <= 2025
                AND `draft_combine_stats`.`lane_agility_time` IS NOT NULL
                AND `draft_combine_stats`.`max_vertical_leap` IS NOT NULL
                AND `draft_combine_stats`.`standing_vertical_leap` IS NOT NULL
                AND `draft_combine_stats`.`player_name` IS NOT NULL
                AND `draft_combine_stats`.`wingspan` IS NOT NULL
                AND `draft_combine_stats`.`weight` IS NOT NULL
                AND `draft_combine_stats`.`height_wo_shoes` IS NOT NULL
                AND `draft_combine_stats`.`bench_press` IS NOT NULL
                {filter_benchpress_zero}
                AND `advanced-stats`.`dbpm` IS NOT NULL  
                AND `advanced-stats`.`mp` >= 200
                AND `advanced-stats`.`pos` IN {pos_placeholder}
            GROUP BY 
                `draft_combine_stats`.`player_name`

        """
        self.models = {}
        self.valid_positions = ["pg", "sg", "sf", "pf", "c"]
        self.models["pg"] = self.createModelGuards()
        self.models["sg"] = self.models["pg"]  # same model for guards
        self.models["sf"] = self.createModelForward()
        self.models["pf"] = self.createModelBigs()
        self.models["c"] = self.models["pf"]  # same model for bigs



    def downloadData(self):
        kaggle_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.path.expanduser("~"), ".kaggle")
        if not os.path.exists(kaggle_path):
            print("ERROR: kaggle.json not found. Please place it in ~/.kaggle/")
            exit(1)
        datasetOne = "wyattowalsh/basketball"
        # datasetTwo = "owenrocchi/nba-advanced-stats-20022022"
        datasetTwo = "sumitrodatta/nba-aba-baa-stats"
        if not os.path.isfile("advancedStats.db"):
            kaggle.api.dataset_download_files(dataset=datasetOne, path=".", unzip=True)

        conn = sqlite3.connect('advancedStats.db')

        if not os.path.isfile("Advanced.csv"):
            kaggle.api.dataset_download_file(dataset=datasetTwo, file_name="Advanced.csv",path=".")
        if os.path.isfile("Advanced.csv"):
            csvFile = pandas.read_csv('Advanced.csv')
            csvFile.to_sql('advanced-stats', conn, if_exists='replace', index=False)
            # Write the DataFrame to a new SQLite table


        conn.commit()
        conn.close()
        os.makedirs("./importances-graphs", exist_ok=True)
        os.makedirs("./scatterplots", exist_ok=True)
        os.makedirs("./shap", exist_ok=True)


    def createModelGuards(self):
        source_conn = sqlite3.connect('nba.sqlite')
        source_conn.execute("ATTACH DATABASE 'advancedStats.db' AS advancedStats_db")

        the_query = self.query.replace("{pos_placeholder}", "('PG', 'SG')")
        the_query = the_query.replace("{filter_benchpress_zero}", "")

        df = pandas.read_sql(the_query, source_conn)
        # print(df.shape)
        dcs_to_z = ['height_wo_shoes', 'weight', 'wingspan', 'standing_vertical_leap', 'lane_agility_time',
                    'three_quarter_sprint', 'max_vertical_leap', 'bench_press']

        df[dcs_to_z] = df[dcs_to_z].apply(pandas.to_numeric, errors='coerce')
        df = df.dropna()
        df['avg_dbpm'] = pandas.to_numeric(df['avg_dbpm'],errors='coerce')
        df[['height_wo_shoes', 'weight']] = df[['height_wo_shoes', 'weight']] ** 2
        df['agility_per_lb'] = df['lane_agility_time']/ df['weight']
        dcs_to_z.append('agility_per_lb')
        df['agility_per_inch'] = df['lane_agility_time'] / df['height_wo_shoes']
        dcs_to_z.append('agility_per_inch')
        df['quickness'] = (df['three_quarter_sprint'] * df['lane_agility_time']) / df['weight']
        dcs_to_z.append('quickness')
        df['perimeter_defense_potential'] = (df['three_quarter_sprint'] * df['lane_agility_time'] * df['wingspan']) / df['weight']
        dcs_to_z.append('perimeter_defense_potential')
        df['speed_per_lb'] = df['three_quarter_sprint']/ df['weight']
        dcs_to_z.append('speed_per_lb')
        df['speed_per_inch'] = df['three_quarter_sprint']/ df['height_wo_shoes']
        dcs_to_z.append('speed_per_inch')
        df['vertical_per_lb'] = df['standing_vertical_leap']/ df['weight']
        dcs_to_z.append('vertical_per_lb')
        df['explosiveness'] =  (df['max_vertical_leap'] * df['three_quarter_sprint']  * df['lane_agility_time'])
        dcs_to_z.append('explosiveness')
        df['jump'] = (df['max_vertical_leap'] * df['standing_vertical_leap'])
        dcs_to_z.append('jump')
        df['size'] = df['height_wo_shoes'] * df['weight']
        df['quickness_over_size'] = (df['quickness']) / df['size']
        dcs_to_z.append('quickness_over_size')
        df['bench_press_over_svertical'] = df['bench_press'] / df['standing_vertical_leap']
        dcs_to_z.append('bench_press_over_svertical')

        df['speed_to_size'] = df['three_quarter_sprint'] / df['size']
        dcs_to_z.append('speed_to_size')
        df['speed_quickness_ratio'] = df['three_quarter_sprint'] / df['quickness']
        dcs_to_z.append('speed_quickness_ratio')
        df['speed_to_jump_efficiency'] = df['three_quarter_sprint'] / (df['max_vertical_leap'])
        dcs_to_z.append('speed_to_jump_efficiency')
        df['closeout'] = (df['three_quarter_sprint'] * df['max_vertical_leap']) / (df['height_wo_shoes'])
        dcs_to_z.append('closeout')


        X = df[dcs_to_z]
        Y = df['avg_dbpm']
    # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        #Train and fit model
        model = RandomForestRegressor(random_state=42, n_estimators=300, max_depth = 3, min_samples_leaf=2, min_samples_split=5)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)  # Prediction for training set
        y_test_pred = model.predict(X_test)  # Prediction for test set

        # Correct evaluation
        train_r2 = r2_score(y_train, y_train_pred)  # R² on training set
        test_r2 = r2_score(y_test, y_test_pred)  # R² on test set
        mse = mean_squared_error(y_test, y_test_pred)  # MSE on test set

        print(f"Train R^2 Score: {train_r2:.3f}")
        print(f"Test R^2 Score: {test_r2:.3f}")
        print(f"Mean Squared Error: {mse:.3f}")
        arr = [train_r2, test_r2, mse]


        # Train the model
        return {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test_pred": y_test_pred,
            "features": dcs_to_z,
            "train-test-r2_mse": arr
        }



    def createModelForward(self):
        source_conn = sqlite3.connect('nba.sqlite')
        source_conn.execute("ATTACH DATABASE 'advancedStats.db' AS advancedStats_db")

        the_query = self.query.replace("{pos_placeholder}", "('SF')")
        the_query = the_query.replace("{filter_benchpress_zero}", "AND `draft_combine_stats`.`bench_press` != 0")
        # the_query = the_query.replace("{filter_benchpress_zero}", "")

        df = pandas.read_sql(the_query, source_conn)
        # print(df.shape)
        # df_two = pandas.read_parquet('advanced.parq')
        # pandas.set_option('display.max_rows', None)  # Show all rows
        # pandas.set_option('display.max_columns', None)  # Show all columns
        # pandas.set_option('display.width', None)  # Disable line wrapping
        # print(df_two)
        dcs_to_z = ['height_wo_shoes', 'weight', 'wingspan', 'standing_vertical_leap', 'lane_agility_time',
                    'three_quarter_sprint', 'max_vertical_leap', 'bench_press']
        df[dcs_to_z] = df[dcs_to_z].apply(pandas.to_numeric, errors='coerce')
        df = df.dropna()
        df['avg_dbpm'] = pandas.to_numeric(df['avg_dbpm'],errors='coerce')
        # df[['height_wo_shoes', 'weight']] = df[['height_wo_shoes', 'weight']] ** 2
        df['quickness'] = df['three_quarter_sprint'] * df['lane_agility_time']
        dcs_to_z.append('quickness')
        # df['size'] = df['height_wo_shoes'] * df['weight']
        df['interior_defense_potential'] =  (df['standing_vertical_leap'] * df['wingspan'] * df['height_wo_shoes'] * df['weight'])
        dcs_to_z.append('interior_defense_potential')
        df['standing_block_potential'] = (df['standing_vertical_leap'] * df['wingspan']  * df['lane_agility_time'])
        dcs_to_z.append('standing_block_potential')
        df['agility_per_wingspan'] = df['lane_agility_time'] / df['wingspan']
        dcs_to_z.append('agility_per_wingspan')
        df['speed_per_wingspan'] = df['three_quarter_sprint'] / df['wingspan']
        dcs_to_z.append('speed_per_wingspan')



        # self.original_means_sf = df[dcs_to_z].mean()  # computed on the original data
        # self.original_stds_sf = df[dcs_to_z].std()  # computed on the original data
        # df[dcs_to_z] = (df[dcs_to_z] - self.original_means_sf) / self.original_stds_sf

        X = df[dcs_to_z]
        Y = df['avg_dbpm']
    # Split data into training and testing sets

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # model = LinearRegression()
        model = RandomForestRegressor(random_state=42, n_estimators=375, max_depth = 3, min_samples_leaf=2, min_samples_split=2)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)  # Prediction for training set
        y_test_pred = model.predict(X_test)  # Prediction for test set

        # Correct evaluation
        train_r2 = r2_score(y_train, y_train_pred)  # R² on training set
        test_r2 = r2_score(y_test, y_test_pred)  # R² on test set
        mse = mean_squared_error(y_test, y_test_pred)  # MSE on test set

        print(f"Train R^2 Score: {train_r2:.3f}")
        print(f"Test R^2 Score: {test_r2:.3f}")
        print(f"Mean Squared Error: {mse:.3f}")
        # print(df['avg_dbpm'].describe())

        arr = [train_r2, test_r2, mse]
        # print("MEANS:\n", self.original_means_sf)
        # print("STDS:\n", self.original_stds_sf)
        # print("RAW INPUT BEFORE SCALING:\n", df)

        return {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test_pred": y_test_pred,
            "features": dcs_to_z,
            "train-test-r2_mse": arr
        }



    def createModelBigs(self):
        source_conn = sqlite3.connect('nba.sqlite')
        source_conn.execute("ATTACH DATABASE 'advancedStats.db' AS advancedStats_db")

        the_query = self.query.replace("{pos_placeholder}", "('PF', 'C')")
        the_query = the_query.replace("{filter_benchpress_zero}", "")

        df = pandas.read_sql(the_query, source_conn)
        # print(df.shape)

        dcs_to_z = ['height_wo_shoes', 'weight', 'wingspan', 'standing_vertical_leap', 'lane_agility_time',
                    'three_quarter_sprint', 'max_vertical_leap', 'bench_press']
        df[dcs_to_z] = df[dcs_to_z].apply(pandas.to_numeric, errors='coerce')
        df['avg_dbpm'] = pandas.to_numeric(df['avg_dbpm'],errors='coerce')
        df[['height_wo_shoes', 'weight']] = df[['height_wo_shoes', 'weight']]  ** 2

        df['agility_per_lb'] = df['lane_agility_time']/ df['weight']
        dcs_to_z.append('agility_per_lb')
        df['agility_per_inch'] = df['lane_agility_time']/ df['height_wo_shoes']
        dcs_to_z.append('agility_per_inch')
        df['vertical_per_lb'] = df['standing_vertical_leap']/ df['weight']
        dcs_to_z.append('vertical_per_lb')
        df['vertical_per_inch'] = df['standing_vertical_leap'] / df['height_wo_shoes']
        dcs_to_z.append('vertical_per_inch')
        df['interior_defense_potential'] =  (df['standing_vertical_leap'] * df['wingspan'] * df['height_wo_shoes']) / df['weight']
        dcs_to_z.append('interior_defense_potential')
        df['standing_block_potential'] = (df['standing_vertical_leap'] * df['wingspan']  * df['lane_agility_time'])/ df['weight']
        dcs_to_z.append('standing_block_potential')
        df['moving_block_potential'] = (df['max_vertical_leap'] * df['wingspan']  * df['lane_agility_time'])/ df['weight']
        dcs_to_z.append('moving_block_potential')

        X = df[dcs_to_z]
        Y = df['avg_dbpm']
    # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize the decision tree regressor
        # Train the model
        model = RandomForestRegressor(random_state=42, n_estimators=375, max_depth = 3, min_samples_leaf=2, min_samples_split=2)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)  # Prediction for training set
        y_test_pred = model.predict(X_test)  # Prediction for test set

        # Correct evaluation
        train_r2 = r2_score(y_train, y_train_pred)  # R² on training set
        test_r2 = r2_score(y_test, y_test_pred)  # R² on test set
        mse = mean_squared_error(y_test, y_test_pred)  # MSE on test set

        print(f"Train R^2 Score: {train_r2:.3f}")
        print(f"Test R^2 Score: {test_r2:.3f}")
        print(f"Mean Squared Error: {mse:.3f}")

        arr = [train_r2, test_r2, mse]

        # df[['weight', 'height_wo_shoes']] = df[['weight', 'height_wo_shoes']] ** .5
        return {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test_pred": y_test_pred,
            "features": dcs_to_z,
            "train-test-r2_mse": arr
        }



    def generateScatterplotsOnTestandTrainSet(self, pos):
        if pos.lower() not in self.valid_positions:
            print("Invalid position")
            return 1
        model = self.models[pos]["model"]
        features = self.models[pos]["features"]
        X_train = self.models[pos]["X_train"]
        X_test = self.models[pos]["X_test"]
        y_train = self.models[pos]["y_train"]
        y_test_pred = self.models[pos]["y_test_pred"]
        plot_dir = os.path.join("scatterplots", pos)
        os.makedirs(plot_dir, exist_ok=True)
        for feature in features:
            fig, ax = pyplot.subplots(1, 2, figsize=(12, 4))

            # Feature vs Actual DBPM (training set)
            ax[0].scatter(X_train[feature], y_train, alpha=0.6, color='blue')
            ax[0].set_title(f"{feature} vs Actual DBPM")
            ax[0].set_xlabel(feature)
            ax[0].set_ylabel("Actual DBPM")

            # Feature vs Predicted DBPM (test set)
            ax[1].scatter(X_test[feature], y_test_pred, alpha=0.6, color='green')
            ax[1].set_title(f"{feature} vs Predicted DBPM")
            ax[1].set_xlabel(feature)
            ax[1].set_ylabel("Predicted DBPM")

            pyplot.tight_layout()
            filename = os.path.join(plot_dir, f"{feature}_scatter.png")
            if not os.path.isfile(filename):
                pyplot.savefig(filename)
            pyplot.clf()
            pyplot.close()

        print(f"📊 Scatter plots saved to folder: {plot_dir}/")

    def generateFeatureImportanceGraph(self, pos):
        if pos.lower() not in self.valid_positions:
            print("Invalid position")
            return 1
        model = self.models[pos]["model"]
        dcs_to_z = self.models[pos]["features"]
        features_importance = model.feature_importances_
        sorted_ind = numpy.argsort(features_importance)[::-1]
        features = numpy.array(dcs_to_z)
        pyplot.bar(features[sorted_ind], features_importance[sorted_ind])
        pyplot.xticks(rotation=45)
        pyplot.title("Feature Importances in Model")
        filename = os.path.join("importances-graphs", f"feature-importances_{pos}.png")
        if not os.path.isfile(filename):
            print(f"📊 importances graph saved as {filename} ✅")
            pyplot.savefig(filename)
        pyplot.show()



    def makePrediction(self, pos, player_data):
        if pos not in self.valid_positions:
            print("Invalid position")
            return 1
        player_data = [float(i) for i in player_data]
        player_data_df= self.featureEngineerBasedOnPosition(pos, player_data)

        model = self.models[pos]["model"]
        results = model.predict(player_data_df)

        return results

    def engineer_features(self, raw_inputs, pos):
        # Unpack inputs
        h, w, ws, svl, agility, sprint, max_vert, bench = raw_inputs

        h2 = h
        w2 = w
        if pos not in ['sf']:
            h2 = h ** 2
            w2 = w ** 2


        # Compute everything (even if not used for all roles)
        features = {
            'height_wo_shoes': h2,
            'weight': w2,
            'wingspan': ws,
            'standing_vertical_leap': svl,
            'lane_agility_time': agility,
            'three_quarter_sprint': sprint,
            'max_vertical_leap': max_vert,
            'bench_press': bench,

            'agility_per_lb': agility / w,
            'agility_per_inch': agility / h,
            'agility_per_wingspan': agility / ws,
            'quickness': sprint * agility / w,
            'quickness_sf': sprint * agility,
            'perimeter_defense_potential': sprint * agility * ws / w,
            'speed_per_lb': sprint / w,
            'speed_per_inch': sprint / h,
            'speed_per_wingspan': sprint / ws,
            'vertical_per_lb': svl / w,
            'vertical_per_inch': svl / h,
            'explosiveness': max_vert * sprint * agility,
            'jump': max_vert * svl,
            'quickness_over_size': (sprint * agility / w) / (h * w),
            'bench_press_over_svertical': bench / svl,
            'speed_to_size': sprint / (h * w),
            'speed_quickness_ratio': sprint / (sprint * agility / w),
            'speed_to_jump_efficiency': sprint / (max_vert * svl),
            'closeout': (sprint * max_vert) / h,
            'interior_defense_potential': svl * ws * h * w,
            'standing_block_potential': svl * ws * agility,
            'moving_block_potential': max_vert * ws * agility
        }

        return features

    def featureEngineerBasedOnPosition(self, pos, player_data_arr):
        if pos not in self.valid_positions:
            print("Invalid position")
            return 1
        full_features = self.engineer_features(player_data_arr, pos)
        pos_features = self.models[pos]["features"]
        resulting_features = [full_features.get(feature) for feature in pos_features]
        df = pandas.DataFrame([resulting_features], columns=pos_features)
        # if pos == "sf":
        #     df = (df - self.original_means_sf[pos_features]) / self.original_stds_sf[pos_features]
        df.columns = df.columns.astype(str)
        # print(df)
        # print("Mean of row:", df.mean(axis=1))
        # print("Max of row:", df.max(axis=1))
        # print("Min of row:", df.min(axis=1))
        return df



    def generateShapData(self, pos, graph_option):
        if pos not in self.valid_positions:
            print("Invalid position")
            return 1
        model = self.models[pos]["model"]
        X_train = self.models[pos]["X_train"]

        explainer = shap.TreeExplainer(model, X_train)
        shap_values = explainer(X_train)

        if graph_option == 0:
            filename = os.path.join("shap", f"shap_bar_{pos}.png")
            shap.plots.bar(shap_values, show=False)
            pyplot.tight_layout()
            if not os.path.isfile(filename):
                pyplot.savefig(filename)
                print(f"📊 SHAP bar chart saved as {filename} ✅")

            pyplot.show()

        elif graph_option == 1:
            filename = os.path.join("shap", f"shap_beeswarm_{pos}.png")
            shap.plots.beeswarm(shap_values, show=False)
            pyplot.tight_layout()
            if not os.path.isfile(filename):
                pyplot.savefig(filename)
                print(f"📊 SHAP beeswarm chart saved as {filename} ✅")

            pyplot.show()

        else:
            print("Invalid graph option")

        pyplot.clf()




