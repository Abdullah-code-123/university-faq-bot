import os
from openai import AsyncOpenAI, OpenAI
from fastapi import FastAPI,HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel,ValidationError,Field
import logging
from sentence_transformers import SentenceTransformer
import time
load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

data=[
   
    "Students must maintain at least 75 percent attendance in each course to be eligible to appear in the final semester examination.",

    "Course registration opens one week before the beginning of each semester and students must register through the university student portal before the deadline.",

    "Semester tuition fees must be paid before the announced due date. Late payments may result in financial penalties or suspension of course registration.",

    "Scholarship applications are submitted through the university scholarship portal. Students must meet the required academic criteria and submit all supporting documents before the application deadline.",

    "The university library is open from 8:00 AM to 8:00 PM on weekdays. Students can borrow books using their university ID card and must return them before the due date.",

    "Hostel accommodation is available for eligible students. Applications can be submitted online after admission confirmation, and room allocation depends on availability.",

    "Students can request official transcripts through the registrar's office by submitting an online application and paying the required processing fee.",

    "Examination schedules are published on the university website before the start of final exams. Students are responsible for checking their exam dates and venues.",

    "Students who wish to withdraw from a course must submit a withdrawal request before the official deadline. Withdrawals after the deadline require approval from the academic department.",

    "The university provides academic advising services to help students select courses, plan their degree requirements, and resolve academic issues.",

    "Students can reset their student portal password by using the password recovery option or by contacting the university IT support center.",

    "The university provides free campus Wi-Fi for students. Access requires a valid university email address and password.",

    "Parking permits are required for students who wish to park vehicles on campus. Permits can be purchased through the university transportation office.",

    "Graduation applications must be submitted during the final semester. Students must complete all degree requirements before they are eligible to graduate.",

    "International students must maintain a valid student visa and follow all immigration regulations throughout their period of study.",

    "The career services office assists students with resume writing, internship opportunities, interview preparation, and job placement support.",

    "Students can report cases of harassment, discrimination, or misconduct through the university's student affairs office. All reports are handled confidentially.",

    "The university health center provides basic medical consultations, first aid, and health awareness services for enrolled students during working hours.",

    "Students who miss an examination due to a valid medical emergency must submit medical documentation to request a makeup examination.",

    "The registrar's office is responsible for maintaining student academic records, issuing enrollment certificates, and processing transcript requests."
]
embeddings=model.encode(data)

class ioquestion(BaseModel):
    query:str=Field(min_length=5,max_length=200)

class oquestion(BaseModel):
    answer:str
    relatedInstruction:str
class AdditionalExplanation(BaseModel):
    instructionExplanation:str
class FinalResponse(BaseModel):
    answer:str
    relatedInstruction:str
    instructionExplanation:str



app=FastAPI()
count_requests=0
window_time = time.time() 
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
    )
@app.post("/InputQuery",response_model=FinalResponse)
async def enterQuery(q:ioquestion): 
    global count_requests,window_time
    if time.time() - window_time > 60:
        count_requests=1
        window_time = time.time()
    count_requests+=1

    if count_requests>=10:
        raise HTTPException(status_code=429,detail="Too many request try again after some time")
    
    blocked=[
    "ignore previous instructions",
    "system prompt",
    "act as",
    "pretend to be"
    ]
    lower = q.query.lower()
    if any(x in lower for x in blocked):
        logger.warning(f'suspicious prompt {ioquestion.query}')
        raise HTTPException(status_code=400,detail="suspicious prompt found")
    embeddings2=model.encode(q.query)
    simi=model.similarity(embeddings,embeddings2)

    logger.info(f'simiilarity is: {simi}')
    best_chunk=simi.argmax().item()
    contextchunk = simi.max().item()
    if contextchunk<0.4:
        context_text="question is irrelvant and has no major connection with context"
    else:
        context_text=data[best_chunk]

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":f"""
             <role>you are an agent that gives the answer of the query asked by students(FAQ)</role>
             <instruction>Donot give the answer straightfarward analyze the <context> tag and answer accordingly</instruction>
             <context>
             {context_text}
             </context>
             <output_format>
             answer should be in the valid json format
             with this pattern
             "answer":"...."
             "relatedInstruction":"..."
             </output_format>
             
             """} ,
            {"role":"user","content":f"""
             <question>
             {q.query}
             </question>
             """}
        ]
    )
    raw =  response.choices[0].message.content
    try:
        result=oquestion.model_validate_json(raw)
    except (ValidationError) as e:
        raise HTTPException(status_code=400,detail=str(e))
    response2=await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":"""
             <context>take {result.RelatedInstruction} as context</context>
             <role>your job is to explain the instruction in such a way that it is easy for a layman to undertsnd</role>
             <instruction>use the <context> tag and answer in a valid json format</instruction>
             <output_format>
             "instructionExplanation":"...."
             </output_format>
             """},
             {"role":"user","content":"""
              
              <question>
              {result.RelatedInstruction}
              </question>
              
              """}
        ]
    )
    raw2 = response2.choices[0].message.content
    try:
        result2=AdditionalExplanation.model_validate_json(raw2)
    except (ValidationError) as e:
        raise HTTPException(status_code=400,detail=str(e))
    
    return FinalResponse(
    answer=result.answer,
    relatedInstruction=result.relatedInstruction,
    instructionExplanation=result2.instructionExplanation
    )


