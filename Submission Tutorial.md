## e-CARE Submission Tutorial

We use [CodaLab Worksheets](http://worksheets.codalab.org/) for submitting your model to get the results on the test set, and put the results onto the leaderboard. 

If your want to get the results on the e-CARE test set, you should follow this instruction.


### Step 1: Get Ready

For running your model, you should package the environment of your model into a Docker image and upload it to [Docker Hub](https://hub.docker.com/).



### Step 2: Upload Necessary Files

* Firstly, you should create a new worksheet in [CodaLab Worksheets](http://worksheets.codalab.org/).
* Secondly, upload necessary files (eg. trained model, prediction script, model framework, e-CARE dev set, evaluation script) via the `UPLOAD` button on the CodaLab Worksheet menu bar. (Compressing all the files into a zip file is recommended, CodaLab would unzip the file automatically once the upload is done.)



### Step 3: Get Results on e-CARE Dev Set with Official Evaluation Script

* Firstly, create a new run through the `RUN` button on the CodaLab Worksheet menu bar.

* Secondly, add the dependencies and fill in the necessary information~(eg. name).

* Thirdly, fill the Docker imagine name in `Step 1` and choose the resources for computing.

* Fourthly, type the prediction command in the `Command` line (eg. `python prediction.py dev.jsonl prediction.json`) for running prediction on dev set. Once successful, a new term will be produced on the worksheet.  And The format of the prediction file should be as follows:

  * `Causal Reasoning`: each key is the `index` of the corresponding example, each value is the prediction label `0` or `1`.

  ```json
  {
    "dev-0": 0,
    "dev-1": 1,
    "dev-2": 0
  }
  ```

  * `Conceptual Explanation Generation`: each key is the `index` of the corresponding example, each value is the generated conceptual explanation.

  ```json
  {
    "dev-0": "Copper is a good thermal conductor.",
    "dev-1": "Abalone are one of the first food items taken by otters as they move into new habitat.",
    "dev-2": "Deserts are arid environments."
  }
  ```

* Finally, In order to unify the format of input and output, you should use the official evaluation scripts for get the evaluation metrics (create a new `Run`, and type `python causal_reasoning.py prediction.json dev.jsonl` in the command line.). We provide two kinds of evaluation scripts:

  * `causal_reasoning.py`: the script for obtaining evaluation metrics on causal reasoning task

  * `conceptual_explanation_generation.py`: the script for obtaining evaluation metrics on conceptual explanation generation task



### Step 4: Submit

Once you get the evaluation metrics on the dev set, you can submit your model to kxiong@ir.hit.edu.cn or ldu@hit.edu.cn, and the following terms should be included in the email.

* The full uuis of the dev set prediction `RUN`~(the term contains the prediction results on dev set in `Step 3`).
* The name of the submitted model.
* The task of your submitted model. (`Causal Reasoning` or `Conceptual Explanation Generation`)
* The name of your institute~(Optional).
* The URL of your paper~(Optional).

