import sys
import os
# Add main directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from handwriting.handwriting import make_prediction as predict_handwriting
from voice.voice import make_prediction as predict_voice_memo
from sts.sts import main

'''
Create the progression score by combining the predictions from the three models

Returns:
    progression_score (float): The progression score
'''
def create_progression_score(handwriting_image, voice_memo, video, medication_status, dbs_status):
    # Run all three independent models and get the predictions
    handwriting_predictions = predict_handwriting(handwriting_image)
    voice_memo_predictions = predict_voice_memo(voice_memo)
    sts_predictions = main(video, medication_status, dbs_status)

    # Define weights for each model
    handwriting_weight = 0.3
    voice_memo_weight = 0.3
    sts_weight = 0.4

    # Print the predictions
    print('Handwriting Prediction:', handwriting_predictions[1])
    print('Voice Memo Prediction:', voice_memo_predictions[0][0])
    print('STS Prediction:', sts_predictions[0][0])

    # Calculate the progression score
    progression_score = (handwriting_weight * handwriting_predictions[1] +
                         voice_memo_weight * (1 - voice_memo_predictions[0][0]) +
                         sts_weight * (1 - sts_predictions[0][0]))
    
    return progression_score