import numpy as np
import pandas as pd
from ibmmpy.src.ibmmpy.ibmm import EyeClassifier

def get_gaze_deviation_from_head(gaze_x, gaze_y, gaze_z):
    # generates pitch and yaw angles of gaze ray from head direction
    # head direction is (1,0,0)
    yaw = np.arctan2(gaze_y, gaze_x)
    pitch = np.arctan2(gaze_z, gaze_x)
    
    return yaw*180/np.pi, pitch*180/np.pi

def add_gaze_event2df(df_new):    
    '''
    Given a data_df with gaze points add a column for gaze events (fixations, saccades, noise)
    '''
    
    df2 = df_new.copy()

    # add approx head compensation
    df2['Cgaze_x'] = df_new.EyeTracker_COMBINEDGazeDir.apply(lambda x: x[0])
    df2['Cgaze_y'] = df_new.EyeTracker_COMBINEDGazeDir.apply(lambda x: x[1])
    df2['Cgaze_z'] = df_new.EyeTracker_COMBINEDGazeDir.apply(lambda x: x[2])

    # gaze+head values
    gaze_pitches, gaze_yaws = get_gaze_deviation_from_head(df2.Cgaze_x, df2.Cgaze_y, df2.Cgaze_z)
    # head_rots = df2.CameraRot.values
    head_pitches = df2.EgoVariables_CameraRot.apply(lambda x: x[0])
    head_yaws = df2.EgoVariables_CameraRot.apply(lambda x: x[2])
    gaze_head_pitches = gaze_pitches + head_pitches
    gaze_head_yaws = gaze_yaws + head_yaws       

    # Create the new pd
    gazeHeadDF = pd.DataFrame(df2[['TimeElapsed']])
    gazeHeadDF = gazeHeadDF.rename(columns={'TimeElapsed':'timestamp'})
    gazeHeadDF['confidence'] = (df2.EyeTracker_LEFTEyeOpennessValid*df2.EyeTracker_RIGHTEyeOpennessValid).astype(bool)
#     gazeHeadDF['x'] = gaze_head_pitches
#     gazeHeadDF['y'] = gaze_head_yaws
#     gazeHeadDF['z'] = np.zeros(len(gaze_head_pitches))
    gazeHeadDF['x'] = df2['Cgaze_x']
    gazeHeadDF['y'] = df2['Cgaze_y']
    gazeHeadDF['z'] = df2['Cgaze_z']

    vel_w = EyeClassifier.preprocess(gazeHeadDF, dist_method="vector")
#     vel_w = EyeClassifier.preprocess(gazeHeadDF, dist_method="euclidean")
    model = EyeClassifier()
    model.fit(world=vel_w)
    # raw_vel = vel_w[np.logical_not(vel_w.velocity.isna())].velocity.values
    # raw_vel[raw_vel > raw_vel.mean() + 3 * raw_vel.std()]
    # print("Velocity Means: ",model.world_model.means_)
    # 0- fix, 1- sacc, -1 ->noise
    labels, indiv_labels = model.predict(world=vel_w)
    labels_unique = labels
    labels_unique.rename(columns={'label': 'gaze_event_label'}, inplace=True)
    
    df_new = df_new.join(labels_unique["gaze_event_label"])
    return df_new    