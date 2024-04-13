from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

text = """
Two years ago we had a reorganization at work in my department, six positions became four. I was one of the two who was given a new job in a different department. I did not want this job, but I didn't want to be unemployed either, so I ended up taking it. My direct manager said she wanted to reduce my stress level, I was overworked, therefore she thought it was a “good idea.” In hindsight it was discrimination.
I wanted other jobs, they were available, but the HR manager said I was not qualified. Which was bullsh*t. All these other jobs had a smaller learning curve than the job I had been given. I had only little experience in the new job. It took a long time to learn. It only increased my stress level.
One of the four remaining employees after the reorganization did not like her job, so after two years she resigned. I thought here's my chance to come back to my old department, so I applied when the job ad was out. What they asked for in the job ad was a perfect match with me and my previous experience; I had 12 years of it. I was so sure I would get the job.
A week went by and I received an email back on my application. It shocked me, it said thanks but no thanks. I wasn't even invited to an interview! When I asked why, HR said they had better candidates. I said show me. It is my legal right.
So they did, they sent me the other applicants resumés. There were three. One declined because she got a better job, the other was probably too expensive, and the third, who also got the job, was the least qualified.
She started last Monday. A young woman, with a good education, but very, very little job experience… and none in our field. Turns out her resumé has some lies on it, too! But they don't know that. She could be fired for this. (People, when you apply for a job, make sure you are honest because the truth is only one click away these days!)
I just know she is going to be gone in less than a year. She is one of many new employees, all younger than the other half of us, who have 25+ years of experience, know this business inside out and have gone to school part-time to better ourselves to also stay competitive in today's business world. We did not let grass grow under our feet.
These new young employees, “so qualified,” ask us “oldies” everything. I have tired of answering their questions, so now I just say, “I'm not qualified enough to answer.” They give me a weird look.
Older employees are far from not qualified; when you are 50 you've never been smarter. I'm taking my employer to court soon, for direct discrimination of age and disability. They lied and said I was not qualified but I will show them who is the smarter one. I'm going to win. You betcha!
"""

summary = summarizer(text, max_length=200, min_length=20, do_sample=False)[0]['summary_text']

print("Original text:")
print(text)
print("\nSummarized text:")
print(summary)