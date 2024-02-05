from flask_cors import CORS, cross_origin
from guardrails import Guard, validators
import openai
from flask import Flask, jsonify
from flask import request
import logging 
logging.getLogger('flask_cors').level = logging.DEBUG

# create flask app
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}}, expose_headers=['Access-Control-Allow-Origin', 'Content-Type'])

@app.route('/', methods=['GET'])
def home():
  return "Hello, World!"

# create a route that returns the guard execution results
@app.route('/validate', methods=['POST', 'OPTIONS'])
@cross_origin()
def validate():
  # extract format string from query parameters
  validator = request.json['validator']
  validatorArgs = request.json['validatorArgs']
  prompt = request.json['prompt']
  num_reasks = int(request.json['num_reasks'] or 0)

  validatorImport = getattr(validators, validator)

  guard = Guard.from_string(
    validators=[validatorImport(**validatorArgs)],
  )

  guard(
    llm_api=openai.chat.completions.create,
    num_reasks=num_reasks,
    prompt=prompt,
  )

  response_obj = {
    "validated_output": guard.history.last.validated_output,
    "failed_validations": guard.history.last.failed_validations.length,
    "passed_num_reasks": num_reasks,
    "raw_output": guard.history.last.raw_outputs,
    "validation": guard.history.last.validator_logs.length,
    "compiled_prompt": guard.history.last.compiled_prompt,
    "compiled_instructions": guard.history.last.compiled_instructions,
  }
  # make a new response object with headers
  # response = app.response_class(
  #   response=jsonify(response_obj),
  #   status=200,
  #   mimetype='application/json',
  #   content_type='application/json',
  #   headers={
  #     'Access-Control-Allow-Origin': '*',
  #     'Content-Type': 'application/json'
  #   }
  # )
  return jsonify(response_obj)

# run the app
app.run(debug=True)

