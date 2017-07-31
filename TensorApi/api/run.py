import tensorflow as tf
import inceptionslimraspi as inceptionslim
from flask import Flask, request, jsonify
from flask_cors import CORS
import crossdomain

app = Flask(__name__)
CORS(app)

app.run(
    host=app.config.get('HOST', 'localhost'),
    port=app.config.get('PORT', 5000)
)

@app.route('/',methods=['POST', 'GET']) #Added 'Options'
@crossdomain(origin='*')
def give_me_tensor():
    file_upload = request.files['file']
    if file_upload:
        tensorPost = tf.convert_to_tensor(file_upload, dtype=tf.float32)
        print("Post: " + tensorPost)
        model = inceptionslim.Inception()
        pred = model.classify(image=tensorPost)
        model.print_scores(probabilities=pred)
        model._write_summary(logdir='/tmp/tensorflow_logs/example')

    return jsonify({'success': True}), 201

