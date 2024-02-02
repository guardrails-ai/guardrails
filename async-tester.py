from guardrails import Guard, validators
import openai
from flask import Flask
from flask import request

# create flask app
app = Flask(__name__)

# create a route that returns the guard execution results
@app.route('/validate', methods=['GET'])
def validate():
  # extract format string from query parameters
  format_string = request.args.get('format')
  prompt = request.args.get('prompt')
  num_reasks = int(request.args.get('num_reasks') or 0)
  output = f"""<output type='string' format='{format_string}' on_fail='reask'/> """

  guard = Guard.from_string(
    validators=[validators.RegexMatch('W.*', on_fail='reask', )],
    reask_prompt='Please begin the answer with the letter "W".' + prompt,
    reask_instructions='Please begin the answer with the letter "W".',
  )
  #   f"""
  #   <rail version="0.1">
  #     <output type='string' format="{format_string}" on_fail='reask'/>
  #     <prompt>{prompt}</prompt>
  #   </rail>
  #   """
  # )

  guard(
    llm_api=openai.chat.completions.create,
    num_reasks=num_reasks,
    prompt=prompt,
  )


  return {
    "validated_output": guard.history.last.validated_output,
    "failed_validations": guard.history.last.failed_validations.length,
    "passed_num_reasks": num_reasks,
    "raw_output": guard.history.last.raw_outputs,
    "validation": guard.history.last.validator_logs.length,
    "output_style": output,
    "compiled_prompt": guard.history.last.compiled_prompt,
    "compiled_instructions": guard.history.last.compiled_instructions,
  }

# run the app
app.run(debug=True)

