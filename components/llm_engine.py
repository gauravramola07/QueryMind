# components/llm_engine.py
# ============================================
# LLM ENGINE COMPONENT
# Groq AI Integration (Primary)
# Google Gemini (Backup)
# ============================================

import os
import sys
import re
import time
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────
# GLOBAL CLIENT
# ─────────────────────────────────────────────
_groq_client = None


# ─────────────────────────────────────────────
# SETUP GROQ
# ─────────────────────────────────────────────

def setup_llm():
    """
    Initialize Groq AI Client
    
    Returns:
        dict: {
            'success': True/False,
            'model': Groq client or None,
            'error': error message or None
        }
    """
    global _groq_client
    
    try:
        # ── Get API key ───────────────────────────
        api_key = config.GROQ_API_KEY
        
        if not api_key or api_key == "gsk_YOUR_GROQ_KEY_HERE":
            return {
                'success': False,
                'model': None,
                'error': "❌ Groq API key not found. Please add GROQ_API_KEY to your .env file."
            }
        
        # ── Create Groq client ────────────────────
        _groq_client = Groq(api_key=api_key)
        
        print(f"✅ Groq AI initialized successfully!")
        print(f"📦 Model: {config.LLM_MODEL}")
        
        return {
            'success': True,
            'model': _groq_client,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'model': None,
            'error': f"❌ Error setting up Groq: {str(e)}"
        }


# ─────────────────────────────────────────────
# CORE FUNCTION: Call Groq API with Auto-Failover
# ─────────────────────────────────────────────

def call_groq(prompt, system_prompt=None):
    """
    Make a call to Groq API with automatic failover.
    If the primary model hits a rate limit, it instantly falls back
    to the faster, high-limit 8B model.
    """
    global _groq_client
    
    if _groq_client is None:
        return None
    
    # Define our models
    primary_model = config.LLM_MODEL # Usually 'llama-3.3-70b-versatile'
    fallback_model = config.LLM_FALLBACK_MODEL   # Fallback model if primary fails
    
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        # ── ATTEMPT 1: Try Primary Model (70B) ──
        chat_completion = _groq_client.chat.completions.create(
            messages=messages,
            model=primary_model,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        error_msg = str(e).lower()
        
        # ── ATTEMPT 2: Fallback if Rate Limited (429) ──
        if '429' in error_msg or 'rate limit' in error_msg:
            print(f"⚠️ Primary model {primary_model} rate limited. Switching to fallback {fallback_model}...")
            try:
                chat_completion_fallback = _groq_client.chat.completions.create(
                    messages=messages,
                    model=fallback_model, # Using the backup model!
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                )
                return chat_completion_fallback.choices[0].message.content
            except Exception as fallback_error:
                print(f"❌ Fallback API error: {str(fallback_error)}")
                raise fallback_error
        else:
            # If it's a different kind of error (like no internet), raise it normally
            print(f"❌ Groq API error: {str(e)}")
            raise e


# ─────────────────────────────────────────────
# MAIN FUNCTION 1: Generate SQL Query
# ─────────────────────────────────────────────

def generate_sql_query(question, schema, model, chat_history=None):
    """
    Convert natural language question to SQL query
    
    Args:
        question: User's question in plain English
        schema: Database schema description
        model: Groq client (passed for compatibility)
        chat_history: Previous conversation (optional)
    
    Returns:
        dict: {
            'success': True/False,
            'sql_query': generated SQL or None,
            'explanation': what the query does,
            'error': error message or None,
            'response_type': 'sql' or 'text'
        }
    """
    try:
        # ── System prompt ─────────────────────────
        system_prompt = """You are an expert Business Intelligence SQL analyst.
Your job is to convert natural language questions into accurate SQLite SQL queries.

STRICT RULES:
1. ONLY generate SELECT queries (never INSERT, UPDATE, DELETE, DROP, CREATE)
2. Always use exact table and column names from the schema provided
3. Use SQLite-compatible syntax only
4. Add LIMIT 1000 if no limit specified
5. Use ROUND(value, 2) for decimal numbers
6. Use meaningful column aliases

RESPONSE FORMAT - Follow this EXACTLY:
For SQL questions:
TYPE: SQL
SQL: [your complete SQL query]
EXPLANATION: [one sentence explaining what this query does]

For conversational/non-data questions:
TYPE: TEXT
RESPONSE: [your helpful text response]"""

        # ── Build user prompt ─────────────────────
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            chat_context = "\nPREVIOUS CONVERSATION:\n"
            for item in chat_history[-3:]:
                chat_context += f"User: {item.get('question', '')}\n"
                chat_context += f"Assistant: {item.get('answer_summary', '')}\n"
        
        user_prompt = f"""DATABASE SCHEMA:
{schema}
{chat_context}
CURRENT QUESTION: {question}

Generate the appropriate response following the exact format specified:"""

        print(f"🤖 Sending question to Groq AI...")
        print(f"❓ Question: {question}")
        
        # ── Call Groq ─────────────────────────────
        response_text = call_groq(user_prompt, system_prompt)
        
        print(f"📝 Groq response received")
        
        # ── Parse response ────────────────────────
        parsed = parse_llm_response(response_text)
        
        return parsed
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ LLM Error: {error_msg}")
        
        if 'rate' in error_msg.lower() or '429' in error_msg:
            return {
                'success': False,
                'sql_query': None,
                'explanation': None,
                'error': "⚠️ Rate limit reached. Please wait a moment and try again.",
                'response_type': 'error'
            }
        
        return {
            'success': False,
            'sql_query': None,
            'explanation': None,
            'error': f"❌ AI Error: {str(e)}",
            'response_type': 'error'
        }


# ─────────────────────────────────────────────
# MAIN FUNCTION 2: Generate Text Response
# ─────────────────────────────────────────────

def generate_text_response(question, schema, data_context, model):
    """
    Generate a text explanation or insight
    
    Args:
        question: User's question
        schema: Database schema
        data_context: Query results as string
        model: Groq client (for compatibility)
    
    Returns:
        dict: {
            'success': True/False,
            'response': text response,
            'error': error message or None
        }
    """
    try:
        system_prompt = """You are a Business Intelligence Analyst assistant.
Provide clear, concise business insights from data.
Use professional language suitable for executives.
Keep responses focused and actionable."""

        prompt = f"""
Dataset Schema:
{schema}

User Question: "{question}"

Query Results / Data:
{data_context}

Provide:
1. Key Finding (1 sentence)
2. 2-3 bullet point insights
3. One business recommendation

Keep it concise and professional."""
        
        response = call_groq(prompt, system_prompt)
        
        return {
            'success': True,
            'response': response,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'response': None,
            'error': f"❌ Error: {str(e)}"
        }


# ─────────────────────────────────────────────
# MAIN FUNCTION 3: Generate Data Summary
# ─────────────────────────────────────────────

def generate_data_summary(schema, kpis, model):
    """
    Generate executive summary of the dataset
    
    Args:
        schema: Database schema description
        kpis: List of detected KPIs
        model: Groq client (for compatibility)
    
    Returns:
        dict: {
            'success': True/False,
            'summary': executive summary text,
            'error': error message or None
        }
    """
    try:
        kpi_text = "\n".join([
            f"  - {kpi['label']}: {kpi['formatted_value']}"
            for kpi in kpis
        ])
        
        system_prompt = """You are a senior Business Intelligence Analyst.
Create executive summaries that are clear, insightful, and actionable.
Use professional business language."""

        prompt = f"""Analyze this dataset and provide an executive summary.

DATASET SCHEMA:
{schema}

KEY METRICS:
{kpi_text}

Provide:
1. **Dataset Overview** (2-3 sentences)
2. **Key Highlights** (3-4 bullet points)  
3. **Data Quality Assessment** (1-2 sentences)
4. **Recommended Questions to Ask** (3 specific questions)

Maximum 300 words. Professional tone."""
        
        response = call_groq(prompt, system_prompt)
        
        return {
            'success': True,
            'summary': response,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'summary': None,
            'error': f"❌ Error: {str(e)}"
        }


# ─────────────────────────────────────────────
# HELPER FUNCTION 1: Parse LLM Response
# ─────────────────────────────────────────────

def parse_llm_response(response_text):
    """
    Parse the structured response from LLM
    
    Args:
        response_text: Raw text response from Groq
    
    Returns:
        dict: Parsed response
    """
    if not response_text:
        return {
            'success': False,
            'sql_query': None,
            'explanation': None,
            'error': "❌ Empty response from AI",
            'response_type': 'error'
        }
    
    response_text = response_text.strip()
    
    # ── Check for SQL response ────────────────
    if 'TYPE: SQL' in response_text or 'TYPE:SQL' in response_text:
        
        # Extract SQL
        sql_match = re.search(
            r'SQL:\s*(.*?)(?=EXPLANATION:|$)',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        # Extract explanation
        explanation_match = re.search(
            r'EXPLANATION:\s*(.*?)$',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        sql_query = sql_match.group(1).strip() if sql_match else None
        explanation = explanation_match.group(1).strip() if explanation_match else "Query executed"
        
        # Clean SQL
        if sql_query:
            sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r'```\s*', '', sql_query)
            sql_query = sql_query.strip().rstrip(';')
        
        if sql_query:
            return {
                'success': True,
                'sql_query': sql_query,
                'explanation': explanation,
                'error': None,
                'response_type': 'sql'
            }
    
    # ── Check for TEXT response ───────────────
    if 'TYPE: TEXT' in response_text or 'TYPE:TEXT' in response_text:
        
        text_match = re.search(
            r'RESPONSE:\s*(.*?)$',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        text_response = text_match.group(1).strip() if text_match else response_text
        
        return {
            'success': True,
            'sql_query': None,
            'explanation': text_response,
            'error': None,
            'response_type': 'text'
        }
    
    # ── Fallback: Try to find SQL in response ─
    sql_match = re.search(
        r'```sql\s*(.*?)```',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    
    if sql_match:
        sql_query = sql_match.group(1).strip()
        return {
            'success': True,
            'sql_query': sql_query,
            'explanation': "Query generated from your question",
            'error': None,
            'response_type': 'sql'
        }
    
    # ── Final fallback: treat as text ─────────
    return {
        'success': True,
        'sql_query': None,
        'explanation': response_text,
        'error': None,
        'response_type': 'text'
    }


# ─────────────────────────────────────────────
# HELPER FUNCTION 2: Test Connection
# ─────────────────────────────────────────────

def test_llm_connection(model):
    """
    Test if Groq API is working
    
    Args:
        model: Groq client
    
    Returns:
        dict: Test result
    """
    try:
        response = call_groq(
            "Say exactly these words: Groq AI connected successfully!",
            "You are a helpful assistant."
        )
        
        return {
            'success': True,
            'message': f"✅ {response.strip()}"
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f"❌ Connection failed: {str(e)}"
        }