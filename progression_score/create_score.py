from handwriting.handwriting import predict_handwriting
from voice.voice import make_prediction as predict_voice_memo
from sts.pose_model import main

'''
Create the progression score by combining the predictions from the three models

Returns:
    progression_score (float): The progression score
'''
def create_progression_score():
    # Run all three independent models and get the predictions
    handwriting_predictions = predict_handwriting()
    voice_memo_predictions = predict_voice_memo()
    sts_predictions = main()

    # Define weights for each model
    handwriting_weight = 0.3
    voice_memo_weight = 0.3
    sts_weight = 0.4

    # Calculate the progression score
    progression_score = (handwriting_weight * handwriting_predictions[1] +
                         voice_memo_weight * (1 - voice_memo_predictions[0][0]) +
                         sts_weight * (1 - sts_predictions[0][0]))
    
    return progression_score