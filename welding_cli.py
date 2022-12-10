import numpy as np
import pickle
import sys

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor

def main(x):
    '''
    get welding parameters IW, IF, WV, FP from input vector x
    load trained model and scaler from files
    return predicted depth and width of weld
    :return:
    tuple(float, float)
    '''
    #load scaler and model
    try:
        scf = open('scaler_X.pkl', 'rb')
    except IOError:
        print('No "scaler_X.pkl" file')
        return (0,0)
    scaler = pickle.load(scf)
    try:
        mf = open('erfr_model.pkl', 'rb')
    except IOError:
        print('No "erfr_model.pkl" file')
        return (0,0)
    model = pickle.load(mf)

    #scaling input vector
    x_scaled = scaler.transform([x])

    #calculating weld
    prediction = model.predict(x_scaled)
    return prediction[0,0], prediction[0,1]




if __name__ == '__main__':
    X_list_for_predict = []

    #read welding parameters from stdin
    print('\nВведите через пробел  IW, IF, WV, FP: ',end='')
    IW, IF, WV, FP = map(float, sys.stdin.readline().split())

    # building vector for predict
    X_list_for_predict.append(float(IW))
    X_list_for_predict.append(float(IF))
    X_list_for_predict.append(float(WV))
    X_list_for_predict.append(float(FP))

    # call prediction and print result
    depth, width = main(np.array(X_list_for_predict))
    print(f'\nГлубина: {depth.round(4)}\nШирина:  {width.round(4)}')
