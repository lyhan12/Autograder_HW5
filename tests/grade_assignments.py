import unittest

from gradescope_utils.autograder_utils.decorators import weight, number, partial_credit

import pandas as pd
import numpy as np

from cell_utils import (
        register_local_file, 
        extract_variables, 
        extract_initial_variables, 
        extract_cell_content_and_outputs,
        find_cells_with_text, 
        find_cells_by_indices,
        has_string_in_cell,
        has_string_in_code_cells,
        search_plots_in_extracted_vars,
        search_text_in_extracted_content,
        print_text_and_output_cells,
        print_code_and_output_cells)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline


class GradeAssignment(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(GradeAssignment, self).__init__(*args, **kwargs)
        self.notebook_path = None

    @weight(3.0)
    @number("1.1")
    def test_prepare_train_test_splits(self):
        begin_cells = find_cells_with_text(self.notebook_path, "1.1. Preparation of Training and Testing Sets")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "1.2. Linear Regression: Fit a line for the given dataset")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        X_linear = cell_vars.get("X_linear", None)
        Y_linear = cell_vars.get("Y_linear", None)
        split_data = cell_vars.get("split_data", None)

        exists_split_data = split_data is not None
        print("Exists Split Data:", exists_split_data)

        self.assertTrue(exists_split_data);

        X_linear_train, X_linear_test, Y_linear_train, Y_linear_test = split_data(X_linear, Y_linear)

        result_exists = (X_linear_train is not None) and (X_linear_test is not None) and (Y_linear_train is not None) and (Y_linear_test is not None)
        print("Valid Result Exists:", result_exists)
        self.assertTrue(result_exists);

        len_X = X_linear.shape[0]
        len_X_train = X_linear_train.shape[0]
        len_X_test = X_linear_test.shape[0]

        proper_split = (len_X_train + len_X_test == len_X)
        print("Proper Split:", proper_split)
        print("Length X:", len_X)
        print("Length X Train:", len_X_train)
        print("Length X Test:", len_X_test)
        self.assertTrue(proper_split);

    @partial_credit(6.0)
    @number("1.2")
    def test_linear_regression(self, set_score=None):
        begin_cells = find_cells_with_text(self.notebook_path, "1.2. Linear Regression: Fit a line for the given dataset")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "1.3. Linear Model Evaluation")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        fit_model = cell_vars.get("fit_model", None)
        predict_data = cell_vars.get("predict_data", None)

        exists_fit_model = fit_model is not None
        exists_predict_data = predict_data is not None

        print("exists_fit_model:", exists_fit_model)
        print("exists_predict_data:", exists_predict_data)
        
        if not exists_fit_model or not exists_predict_data:
            set_score(0.0)
            return

        X_linear_train = cell_vars.get("X_linear_train", None)
        Y_linear_train = cell_vars.get("Y_linear_train", None)
        X_linear_test = cell_vars.get("X_linear_test", None)

        linear_model = fit_model(X_linear_train, Y_linear_train)
        Y_train_pred, Y_test_pred = predict_data(linear_model, X_linear_train, X_linear_test)

        proper_linear_model = type(linear_model) == LinearRegression
        print("Type of Linear Model:", type(linear_model))
        print("Proper Linear Model (Linear Regression):", proper_linear_model)


        Y_train_model = linear_model.predict(X_linear_train)
        Y_test_model = linear_model.predict(X_linear_test)

        mse_train = np.mean((Y_train_pred - Y_train_model) ** 2)
        mse_test = np.mean((Y_test_pred - Y_test_model) ** 2)

        proper_prediction = mse_train < 1e-4 and mse_test < 1e-4
        print("Proper Prediction:", proper_prediction)

        score = 0
        if proper_linear_model:
            score += 3.0
        if proper_prediction:
            score += 3.0

        set_score(score)

    @partial_credit(4.0)
    @number("1.3")
    def test_linear_model_eval(self, set_score=None):
        begin_cells = find_cells_with_text(self.notebook_path, "1.3. Linear Model Evaluation")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "1.4. Discussion about the Evaluation Results.")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        evaluate_model = cell_vars.get("evaluate_model", None)
        exists_evaluate_model = evaluate_model is not None
        print("exists_evaluate_model:", exists_evaluate_model)

        Y_linear_train = cell_vars.get("Y_linear_train", None)
        Y_linear_train_pred = cell_vars.get("Y_linear_train_pred", None)
        Y_linear_test = cell_vars.get("Y_linear_test", None)
        Y_linear_test_pred = cell_vars.get("Y_linear_test_pred", None)

        mse_linear_train, mse_linear_test, r2_linear_train, r2_linear_test = evaluate_model(
            Y_linear_train, Y_linear_train_pred, Y_linear_test, Y_linear_test_pred)

        print("MSE from evaluate_model:", mse_linear_train, mse_linear_test)
        print("R2 from evaluate_model:", r2_linear_train, r2_linear_test)

        from sklearn.metrics import mean_squared_error, r2_score
        def evaluate_model_gt(Y_train, Y_train_pred, Y_test, Y_test_pred):
            # YOUR CODE HERE
            mse_train = mean_squared_error(Y_train, Y_train_pred)
            mse_test = mean_squared_error(Y_test, Y_test_pred)
            r2_train = r2_score(Y_train, Y_train_pred)
            r2_test = r2_score(Y_test, Y_test_pred)
            return mse_train, mse_test, r2_train, r2_test

        mse_linear_train_gt, mse_linear_test_gt, r2_linear_train_gt, r2_linear_test_gt = evaluate_model_gt(
                Y_linear_train, Y_linear_train_pred, Y_linear_test, Y_linear_test_pred)

        print("MSE from evaluate_model_gt:", mse_linear_train_gt, mse_linear_test_gt)
        print("R2 from evaluate_model_gt:", r2_linear_train_gt, r2_linear_test_gt)

        proper_mse = abs(mse_linear_train - mse_linear_train_gt) < 1e-4 and abs(mse_linear_test - mse_linear_test_gt) < 1e-4
        proper_r2 = abs(r2_linear_train - r2_linear_train_gt) < 1e-4 and abs(r2_linear_test - r2_linear_test_gt) < 1e-4

        print("Proper MSE Evaluation:", proper_mse)
        print("Proper R2:", proper_r2)

        score = 0
        if proper_mse:
            score += 2.0
        if proper_r2:
            score += 2.0
        set_score(score)

    @partial_credit(0.0)
    @number("1.4")
    def test_linear_model_eval_explanation(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "1.4. Discussion about the Evaluation Results")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "1.5. Implement Polynomial Features")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)

    @partial_credit(8.0)
    @number("1.5")
    def test_poly_feat(self, set_score=None):
        begin_cells = find_cells_with_text(self.notebook_path, "1.5. Implement Polynomial Features")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "Section 2: Linear Regression Model Design")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        fit_model_poly = cell_vars.get("fit_model_poly", None)
        predict_data_poly = cell_vars.get("predict_data_poly", None)

        exists_fit_model_poly = fit_model_poly is not None
        exists_predict_data_poly = predict_data_poly is not None

        print("exists_fit_model_poly:", exists_fit_model_poly)
        print("exists_predict_data_poly:", exists_predict_data_poly)

        degree = 1
        X_train = cell_vars.get("X_linear_train", None)
        X_test = cell_vars.get("X_linear_test", None)
        Y_train = cell_vars.get("Y_linear_train", None)

        model = fit_model_poly(X_train, Y_train, degree)
        Y_train, Y_test = predict_data_poly(model, X_train, X_test)

        proper_type = type(model) == Pipeline
        use_poly_feat = "PolynomialFeatures" in str(model)

        print("Type of Model:", type(model))
        print("Proper Model Type (Pipeline):", proper_type)
        print("Use Polynomial Features:", use_poly_feat)

        def predict_data_poly_gt(model, X_train, X_test):
            # YOUR CODE HERE
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            return Y_train_pred, Y_test_pred

        Y_train_gt, Y_test_gt = predict_data_poly_gt(model, X_train, X_test)
        proper_pred = np.allclose(Y_train, Y_train_gt) and np.allclose(Y_test, Y_test_gt)

        score = 0.0
        if proper_type:
            score += 2.0
        if use_poly_feat:
            score += 4.0
        if proper_pred:
            score += 2.0

        set_score(score)

    @partial_credit(0.0)
    @number("2.1")
    def test_gradient_derivation(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "2.1. Derive the gradients")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "2.2 Implement our SimpleLinearRegression class")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)


    @partial_credit(12.0)
    @number("2.2")
    def test_implement_custom_model(self, set_score=None):
        print('')

        self.assertTrue(True)

        begin_cells = find_cells_with_text(self.notebook_path, "2.2 Implement our SimpleLinearRegression class")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "2.3 Train your model")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)


        model = cell_vars.get('model', None)
        is_defined = model is not None

        SimpleLinearRegression = cell_vars.get('SimpleLinearRegression', None)
        is_complete = type(model) == SimpleLinearRegression

        results = cell_vars.get('results', None)
        answers = cell_vars.get('answers', None)

        good_answer = results == answers

        score = 0.0
        if is_defined:
            score += 3.0
            if is_complete:
                score += 3.0
                if good_answer:
                    score += 6.0

        set_score(score)

    @partial_credit(8.0)
    @number("2.3")
    def test_train_custom_model(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "2.3 Train your model")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "Section 3 : Can Regression do")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)


        model = cell_vars.get('model', None)
        is_defined = model is not None

        SimpleLinearRegression = cell_vars.get('SimpleLinearRegression', None)
        is_complete = type(model) == SimpleLinearRegression

        print("Model Defined:", is_defined)
        print("Model Type:", type(model))
        print("Model Type == SimpleLinearRegression:", is_complete)

        score_1_5_hour = cell_vars.get('score_1_5_hour', None)
        score_3_5_hour = cell_vars.get('score_3_5_hour', None)
        print("Score 1.5 Hour:", score_1_5_hour)
        print("Score 3.5 Hour:", score_3_5_hour)
        right_score_1_5_hour = 40 < score_1_5_hour < 70
        right_score_3_5_hour = 55 < score_3_5_hour < 85

        


        score = 0.0
        if is_defined and is_complete:
            score += 3.0
        if right_score_1_5_hour and right_score_3_5_hour:
            score += 5.0
        set_score(score)

    @partial_credit(0.0)
    @number("2.4")
    def test_exact_linear_regressor(self, set_score=None):
        print('')
        begin_cells = find_cells_with_text(self.notebook_path, "(BONUS) 2.4. Calculate the Exact Solution to the Linear Regressor")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "Section 3 : Can Regression do Classification?")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        print_text_and_output_cells(self.notebook_path, begin_cell_idx, end_cell_idx)


    @weight(2.0)
    @number("3.1")
    def test_theta_init(self):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "##### 3.1")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "##### 3.2 ")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        get_random_vector = cell_vars.get('get_random_vector', None)

        vec1 = get_random_vector()
        vec2 = get_random_vector()

        print("Random Vector?:", vec1 != vec2)
        print("Type of Output:", type(vec1))

        type_numpy = type(vec1) == np.ndarray
        print("Numpy Type Output?:", type_numpy)

        try:
            arr = np.array(vec1)
        except:
            arr = None

        type_numpy_compatible = type_numpy
        if type(arr) == np.ndarray:
            type_numpy_compatible = True
        print("Type is numpy compatible:", type_numpy_compatible)


        if not type_numpy_compatible:
            self.assertTrue(type_numpy_compatible)

        num_el = len(arr)
        print("Number of elements (should be 2):,", num_el)

        if num_el != 2:
            self.assertTrue(num_el == 2)


    @weight(3.0)
    @number("3.3")
    def test_mse_loss_function(self):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "##### 3.3 Write a function for the loss")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "##### 3.4: Training Loop")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        mse_loss = cell_vars.get('mse_loss', None)


        has_mse = mse_loss is not None
        mse_correct = False

        gts = np.array([[1, 2, 3], [0.1, 0.3, 0.5], [10, 20, 30]])
        preds = np.array([[1, 2, 3], [0.05, 0.25, 0.55], [10, 25, 30]])

        mses = np.array([0, 0.0025, 8.3333333333333333333333])
        

        # Check whether mse_loss is working liks this
        # def mse_loss(true_values, predictions):
        #   return  np.mean((true_values- predictions)**2)
        if has_mse:
            if callable(mse_loss):
                count = 0
                for i in range(3):
                    if abs(np.array(mse_loss(gts[i], preds[i])).squeeze().item() - mses[i]) < 1e-4:
                        count += 1
                if count == 3:
                    mse_correct = True

        print("Has mse: ", has_mse)
        print("MSE correct: ", mse_correct)

        result = has_mse and mse_correct

        self.assertTrue(result)
 

    @partial_credit(10.0)
    @number("4.1")
    def test_nn_data_prep(self, set_score=None):
        begin_cells = find_cells_with_text(self.notebook_path, "4.1. Preparation of Data Loaders")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "4.2. Define Multi-Layer Perceptron Model")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        train_loader = cell_vars.get('train_loader', None)
        test_loader = cell_vars.get('test_loader', None)

        from torch.utils.data import DataLoader
        proper_data_loader = type(train_loader) == DataLoader and type(test_loader) == DataLoader
        if not proper_data_loader:
            print("Data Loader Type Wrong:", type(train_loader), type(test_loader))
            set_score(0.0)
            return

        len_train_dataset = len(train_loader)
        len_test_dataset = len(test_loader)
        split_ratio = len_test_dataset / (len_train_dataset + len_test_dataset)
        rubric1_split_ratio = 0.18 < split_ratio < 0.22
        print("len_train_dataset:", len_train_dataset)
        print("len_test_dataset:", len_test_dataset)
        print("split_ratio:", split_ratio)

        batch_size_train = train_loader.batch_size
        batch_size_test = test_loader.batch_size
        print("batch_size_train:", batch_size_train)
        print("batch_size_test:", batch_size_test)
        rubric2_batch_size = batch_size_train == 32 and batch_size_test == 32

        import torch
        if type(train_loader.dataset[0]) == tuple:
            train_dataset = torch.stack([data[0] for data in train_loader.dataset])
        else:
            train_dataset = torch.tensor(train_loader.dataset)
        if type(test_loader.dataset[0]) == tuple:
            test_dataset = torch.stack([data[0] for data in test_loader.dataset])
        else:
            test_dataset = torch.tensor(test_loader.dataset)

        mean_train_dataset = train_dataset.mean(dim=0)
        var_train_dataset = train_dataset.var(dim=0)
        print("mean_train_dataset:", mean_train_dataset)
        print("var_train_dataset:", var_train_dataset)
        mean_test_dataset = test_dataset.mean(dim=0)
        var_test_dataset = test_dataset.var(dim=0)
        print("mean_test_dataset:", mean_test_dataset)
        print("var_test_dataset:", var_test_dataset)

        zero_mean_train = torch.all(abs(mean_train_dataset) < 1e-4)
        unit_var_train = torch.all(abs(var_train_dataset - 1) < 1e-4)
        print("zero_mean_train:", zero_mean_train)
        print("unit_var_train:", unit_var_train)
        use_norm = zero_mean_train and unit_var_train
        rubric3_norm = use_norm

        from torch.utils.data.sampler import RandomSampler
        train_sampler = train_loader.sampler
        use_random_sampler = type(train_sampler) == RandomSampler
        print("Sampler Type:", type(train_sampler))
        print("Use Random Sampler:", use_random_sampler)
        rubric4_random = use_random_sampler 

        batch_size = cell_vars.get('batch_size', None)
        num_train_data = cell_vars.get('num_train_data', None)
        num_test_data = cell_vars.get('num_test_data', None)
        feat_dim = cell_vars.get('feat_dim', None)

        proper_batch_size = batch_size == 32
        proper_num_train_data = num_train_data == 16512
        proper_num_test_data = num_test_data == 4128
        proper_feat_dim = feat_dim == 8

        print("batch_size:", batch_size)
        print("num_train_data:", num_train_data)
        print("num_test_data:", num_test_data)
        print("feat_dim:", feat_dim)
        print("proper_batch_size(32):", proper_batch_size)
        print("proper_num_train_data(16512):", proper_num_train_data)
        print("proper_num_test_data(4128):", proper_num_test_data)
        print("proper_feat_dim(8):", proper_feat_dim)
        exact_all_values = proper_batch_size and proper_num_train_data and proper_num_test_data and proper_feat_dim
        print("All Values Exact:", exact_all_values)


        print("rubric1_split_ratio:", rubric1_split_ratio)
        print("rubric2_batch_size:", rubric2_batch_size)
        print("rubric3_norm:", rubric3_norm)
        print("rubric4_random:", rubric4_random)
        print("rubric5_values:", exact_all_values)

        score = 0.0
        if rubric1_split_ratio:
            score += 2.0
        if rubric2_batch_size:
            score += 2.0
        if rubric3_norm:
            score += 2.0
        if rubric4_random:
            score += 2.0
        if exact_all_values:
            score += 2.0

        set_score(score)

    @partial_credit(5.0)
    @number("4.2")
    def test_nn_model(self, set_score=None):
        begin_cells = find_cells_with_text(self.notebook_path, "4.2. Define Multi-Layer Perceptron Model")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "4.3. Implement the MLP Training Loop")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)



        model = cell_vars.get('model', None)
        MLPModel = cell_vars.get('MLPModel', None)
        proper_model = type(model) == MLPModel
        if not proper_model:
            print("Model Type is Wrong:", type(model))
            set_score(0.0)
            return

        num_layers = cell_vars.get('num_layers', None)
        input_dim = cell_vars.get('input_dim', None)
        output_dim = cell_vars.get('output_dim', None)

        use_ReLU = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, "relu")
        use_Linear = "Linear" in str(model)

        print("Number of Layers(3):", num_layers)
        print("Input Dimension(8):", input_dim)
        print("Output Dimension(1):", output_dim)
        print("Use ReLU:", use_ReLU)
        print("Use Linear:", use_Linear)

        score = 0.0
        if input_dim == 8 and output_dim == 1:
            score += 2.0
            print("Proper Input-Output Dim (+2)")

            if num_layers == 3:
                print("Proper Number of Layers (+1)")
                score += 1.0
                if use_ReLU:
                    print("Use ReLU (+1)")
                    score += 1.0
                    if use_Linear:
                        print("Use Linear (+1)")
                        score += 1.0
        set_score(score) 




    @partial_credit(5.0)
    @number("5.1")
    def test_cnn_data_prep(self, set_score=None):
        begin_cells = find_cells_with_text(self.notebook_path, "B.1. Preparation of Data Loaders")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "B.2. Define Convolutional Neural Netwtork ")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1, begin_cell_idx=begin_cell_idx)
        train_loader = cell_vars.get('train_loader', None)
        test_loader = cell_vars.get('test_loader', None)

        from torch.utils.data import DataLoader
        proper_data_loader = type(train_loader) == DataLoader and type(test_loader) == DataLoader
        if not proper_data_loader:
            print("Data Loader Type Wrong:", type(train_loader), type(test_loader))
            set_score(0.0)
            return

        batch_size_train = train_loader.batch_size
        batch_size_test = test_loader.batch_size
        print("batch_size_train:", batch_size_train)
        print("batch_size_test:", batch_size_test)
        rubric1_batch_size = batch_size_train == 4 and batch_size_test == 4

        from torch.utils.data.sampler import RandomSampler
        train_sampler = train_loader.sampler
        use_random_sampler = type(train_sampler) == RandomSampler
        print("Sampler Type:", type(train_sampler))
        print("Use Random Sampler:", use_random_sampler)
        rubric2_random = use_random_sampler 

        batch_size = cell_vars.get('batch_size', None)
        num_train_data = cell_vars.get('num_train_data', None)
        num_test_data = cell_vars.get('num_test_data', None)
        feat_dim = cell_vars.get('feat_dim', None)

        proper_batch_size = batch_size == 4
        proper_num_train_data = num_train_data == 50000
        proper_num_test_data = num_test_data == 10000
        proper_feat_dim = feat_dim == 3

        print("batch_size:", batch_size)
        print("num_train_data:", num_train_data)
        print("num_test_data:", num_test_data)
        print("feat_dim:", feat_dim)
        print("proper_batch_size(4):", proper_batch_size)
        print("proper_num_train_data(50000):", proper_num_train_data)
        print("proper_num_test_data(10000):", proper_num_test_data)
        print("proper_feat_dim(3):", proper_feat_dim)
        exact_all_values = proper_batch_size and proper_num_train_data and proper_num_test_data and proper_feat_dim
        print("All Values Exact:", exact_all_values)
        rubric3_values = exact_all_values


        print("rubric1_batch_size:", rubric1_batch_size)
        print("rubric2_random:", rubric2_random)
        print("rubric3_values:", exact_all_values)

        score = 0.0
        if rubric1_batch_size:
            score += 1.5
        if rubric2_random:
            score += 1.5
        if rubric3_values:
            score += 2.0

        set_score(score)


    @partial_credit(5.0)
    @number("5.2")
    def test_cnn_model(self, set_score=None):
        begin_cells = find_cells_with_text(self.notebook_path, "##### B.2. Define Convolutional Neural Netwtork")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']
        end_cells = find_cells_with_text(self.notebook_path, "B.3. Implement the ConvNet Training Loop ")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        ref_cells = find_cells_with_text(self.notebook_path, "Bonus Section: Convolutional Neural Networks")
        ref_cell_idx = ref_cells[0]['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1, begin_cell_idx=ref_cell_idx)

        model = cell_vars.get('model', None)
        ConvNetModel = cell_vars.get('ConvNetModel', None)
        proper_model = type(model) == ConvNetModel
        if not proper_model:
            print("Model Type is Wrong:", type(model))
            set_score(0.0)
            return

        num_layers = cell_vars.get('num_layers', None)
        input_ch = cell_vars.get('input_ch', None)
        output_dim = cell_vars.get('output_dim', None)

        use_ReLU = has_string_in_code_cells(self.notebook_path, begin_cell_idx, end_cell_idx, "relu")
        use_conv = "Conv2d" in str(model)
        use_Linear = "Linear" in str(model)

        print("Number of Layers:", num_layers)
        print("Input Channel(3):", input_ch)
        print("Output Dimension(10):", output_dim)
        print("Use ReLU:", use_ReLU)
        print("Use Conv:", use_conv)
        print("Use Linear:", use_Linear)

        proper_dim = input_ch == 3 and output_dim == 10

        score = 0.0
        if proper_dim:
            score += 2.0
        if use_ReLU:
            score += 1.0
        if use_conv:
            score += 1.0
        if use_Linear:
            score += 1.0

        set_score(score) 


