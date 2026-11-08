# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”
"I used the rule based model only"

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.
"This model is trying to classify simple text messages into different moods while accounting for things such as sarcasm and emojis"
**How it works (brief):**  
For the rule based version, describe the scoring rules you created.  
For the ML version, describe how training works at a high level (no math needed).

The scoring rules I created for the rules based version involved adding a point for positive statements and not adding a point for negative statements, mixed statements registered as both positive and negative


## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

There are over 5 new posts in 'SAMPLE_POSTS'

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.

I chose labels based on my intent in the statement I will admit the one statement where I said I love it when I have to take orders could've been labeled as positive by the model

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

I tried adding in mixed feelings and sarcasm into the dataset

**Possible issues with the dataset:**  
Think about imbalance, ambiguity, or missing kinds of language.

Some of my statements could've been ambiguous, there are more positive and negative statements than mixed and neutral, and there might be stronger language missing

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

Positive words add points to the score
negative words after positive words 

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?

Positive and negative words are predictable 

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.
It has difficulty with sarcasm

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

## 5. Evaluation

**How you evaluated the model:**  
Both versions can be evaluated on the labeled posts in `dataset.py`.  
Describe what accuracy you observed.
The rule based model is accurate in regards to negative and positive words
**Examples of correct predictions:**  
Provide 2 or 3 examples and explain why they were correct.
"I am having fun" positive
This was correct because fun is a positive word
"This show is mid"
this is correct because mid is slang used negatively
**Examples of incorrect predictions:**  
Provide 2 or 3 examples and explain why the model made a mistake.  
If you used both models, show how their failures differed.
"Oh great, more work" positive this was incorrect because the statement was meant to be sarcastic and should be negative
"I'm hopeful, but tired" positive this statement should've been mixed
## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

The dataset is relatively small so certain words can slip past it
## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

The model could possibly misinterpret someone's words
## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang  
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only

More data could improve the model's accuracy