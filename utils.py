import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve, accuracy_score, confusion_matrix, average_precision_score



def episode_metrics(model, x, parameters, rhythm, out_message=False):
    """ arg
    
    """
    # Rhythmn predictions
    predictions_qa, predictions_r = model.predict(x)
    # If the estimated prob of AF(column 1) is higher than the estimated prob of non-AF(column 0)
    # than consider the window as an AF window. 
    y_predictions = np.argmax(predictions_r, axis=1)
    y_truth = np.argmax(rhythm, axis=1)
    # Confusion_matrix
    cf = confusion_matrix(y_truth, y_predictions, labels=[0, 1])
    TN, FP, FN, TP = cf.ravel()
    support = TN+FP+FN+TP
    # Sensitivity, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # F1 score
    f1 = metrics.f1_score(y_truth, y_predictions, average=None)
    # selecting f1score for positive case if present
    f1_pos = f1[1] if len(f1) > 1 else f1[0]
    # Area under precision recall curve
    #auprc = metrics.average_precision_score(y_truth, rhythm[:,1])
    if out_message:
        print(pd.DataFrame(cf).rename(columns={0: "Predicted Non-AF", 1:" Predicted AF"}, index={0: "True Non-AF", 1:"True AF"}))                
        print("Sensitivity/Recall: %0.4f" % TPR) 
        print("Specificity: %0.4f" % TNR) 
        print("Precision/PPV: %0.4f" % PPV) 
        print("Negative predictive value/NPV: %0.2f" % NPV) 
        print("False positive rate: %0.4f" % FPR) 
        print("False negative rate: %0.4f" % FNR) 
        print("F1 score: %0.4f" % f1_pos) 
        print('support: ', support)
        print('\n\n\n')
    episode_metrics = [TPR, TNR, PPV, NPV, FPR, FNR, f1_pos, support]
    return episode_metrics


def episode_metrics_singletask(model, x, parameters, rhythm, out_message=False):
    """ arg
    
    """
    # Rhythmn predictions
    predictions_r = model.predict(x)
    # If the estimated prob of AF(column 1) is higher than the estimated prob of non-AF(column 0)
    # than consider the window as an AF window. 
    y_predictions = np.argmax(predictions_r, axis=1)
    y_truth = np.argmax(rhythm, axis=1)
    # Confusion_matrix
    cf = confusion_matrix(y_truth, y_predictions, labels=[0, 1])
    TN, FP, FN, TP = cf.ravel()
    support = TN+FP+FN+TP
    # Sensitivity, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # F1 score
    f1 = metrics.f1_score(y_truth, y_predictions, average=None)
    # selecting f1score for positive case if present
    f1_pos = f1[1] if len(f1) > 1 else f1[0]
    # Area under precision recall curve
    #auprc = metrics.average_precision_score(y_truth, rhythm[:,1])
    if out_message:
        print(pd.DataFrame(cf).rename(columns={0: "Predicted Non-AF", 1:" Predicted AF"}, index={0: "True Non-AF", 1:"True AF"}))                
        print("Sensitivity/Recall: %0.4f" % TPR) 
        print("Specificity: %0.4f" % TNR) 
        print("Precision/PPV: %0.4f" % PPV) 
        print("Negative predictive value/NPV: %0.2f" % NPV) 
        print("False positive rate: %0.4f" % FPR) 
        print("False negative rate: %0.4f" % FNR) 
        print("F1 score: %0.4f" % f1_pos) 
        print('support: ', support)
        print('\n\n\n')
    episode_metrics = [TPR, TNR, PPV, NPV, FPR, FNR, f1_pos, support]
    return episode_metrics


def collecting_individual_metrics(model, x, parameters, rhythm, out_message=False):
    """
    """
    individual_metrics = {}
    for i in np.unique(parameters['ID']):
        # Sub-selecting individuals
        p_indx = np.where(parameters['ID'] == i)[0]
        x_pID = x[p_indx]
        parameters_pID = parameters.iloc[p_indx]
        rhythm_pID = rhythm[p_indx]
        # Rhythmn predictions
        predictions_qa, predictions_r = model.predict(x_pID)
        # If the estimated prob of AF(column 1) is higher than the estimated prob of non-AF(column 0)
        # than consider the window as an AF window. 
        y_predictions = np.argmax(predictions_r, axis=1)
        y_truth = np.argmax(rhythm_pID, axis=1)
        # Confusion_matrix
        cf = confusion_matrix(y_truth, y_predictions, labels=[0, 1])
        TN, FP, FN, TP = cf.ravel()
        support = TN+FP+FN+TP
        # Sensitivity, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # F1 score
        f1 = metrics.f1_score(y_truth, y_predictions, average=None)
        # selecting f1score for positive case if present
        f1_pos = f1[1] if len(f1) > 1 else f1[0]
        # Area under precision recall curve
        #auprc = metrics.average_precision_score(y_test_truth, rhythm_pID[:,1])
        if (TP + FN) > 0:
            individual_metrics[i] = [TPR, np.NaN, np.NaN, FNR, f1_pos, support]
        if (FP + TN) > 0:
            individual_metrics[i] = [np.NaN, TNR, FPR, np.NaN, np.NaN, support]
        if out_message:
            print("PATIENT :", i , '\n')
            print(pd.DataFrame(cf).rename(columns={0: "Predicted Non-AF", 1:" Predicted AF"}, index={0: "True Non-AF", 1:"True AF"}))
            if (TP + FN) > 0:
                print("Sensitivity/Recall: %0.4f" % TPR) 
                print("False negative rate: %0.4f" % FNR) 
                print("F1 score: %0.4f" % f1_pos) 
                print('support: ' , support)
            if (FP + TN) > 0:
                print("Specificity: %0.4f" % TNR) 
                print("False positive rate: %0.4f" % FPR) 
                print('support: ' , support)
    return individual_metrics

def collecting_individual_metrics_singletask(model, x, parameters, rhythm, out_message=False):
    """
    """
    individual_metrics = {}
    for i in np.unique(parameters['ID']):
        # Sub-selecting individuals
        p_indx = np.where(parameters['ID'] == i)[0]
        x_pID = x[p_indx]
        parameters_pID = parameters.iloc[p_indx]
        rhythm_pID = rhythm[p_indx]
        # Rhythmn predictions
        predictions_r = model.predict(x_pID)
        # If the estimated prob of AF(column 1) is higher than the estimated prob of non-AF(column 0)
        # than consider the window as an AF window. 
        y_predictions = np.argmax(predictions_r, axis=1)
        y_truth = np.argmax(rhythm_pID, axis=1)
        # Confusion_matrix
        cf = confusion_matrix(y_truth, y_predictions, labels=[0, 1])
        TN, FP, FN, TP = cf.ravel()
        support = TN+FP+FN+TP
        # Sensitivity, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # F1 score
        f1 = metrics.f1_score(y_truth, y_predictions, average=None)
        # selecting f1score for positive case if present
        f1_pos = f1[1] if len(f1) > 1 else f1[0]
        # Area under precision recall curve
        #auprc = metrics.average_precision_score(y_test_truth, rhythm_pID[:,1])
        if (TP + FN) > 0:
            individual_metrics[i] = [TPR, np.NaN, np.NaN, FNR, f1_pos, support]
        if (FP + TN) > 0:
            individual_metrics[i] = [np.NaN, TNR, FPR, np.NaN, np.NaN, support]
        if out_message:
            print("PATIENT :", i , '\n')
            print(pd.DataFrame(cf).rename(columns={0: "Predicted Non-AF", 1:" Predicted AF"}, index={0: "True Non-AF", 1:"True AF"}))
            if (TP + FN) > 0:
                print("Sensitivity/Recall: %0.4f" % TPR) 
                print("False negative rate: %0.4f" % FNR) 
                print("F1 score: %0.4f" % f1_pos) 
                print('support: ' , support)
            if (FP + TN) > 0:
                print("Specificity: %0.4f" % TNR) 
                print("False positive rate: %0.4f" % FPR) 
                print('support: ' , support)
    return individual_metrics
