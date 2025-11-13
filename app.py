import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
from dotenv import load_dotenv
import sqlite3
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import json
import random
from datetime import date
import tempfile
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import re
import uuid
from pyvis.network import Network
import streamlit.components.v1 as components
import base64
from PIL import Image
import time
import logging
from datetime import timedelta
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

try:
    genai.configure(api_key=os.getenv("GEMINI_API"))
    client = genai.Client(api_key=os.getenv("GEMINI_API"))
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")

class GeminiRateLimiter:
    def __init__(self):
        self.requests_per_minute = 0
        self.last_reset = datetime.now()
        self.circuit_open = False
        self.failure_count = 0

    def make_request(self, model, prompt, max_retries=3):
        if self.circuit_open:
            if datetime.now() - self.last_reset > timedelta(minutes=5):
                self.circuit_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker open - too many failures")
        if datetime.now() - self.last_reset > timedelta(minutes=1):
            self.requests_per_minute = 0
            self.last_reset = datetime.now()
        if self.requests_per_minute >= 8:
            wait_time = 60 - (datetime.now() - self.last_reset).seconds
            time.sleep(wait_time)
            self.requests_per_minute = 0
            self.last_reset = datetime.now()
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                self.requests_per_minute += 1
                self.failure_count = 0
                return response
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif "400" in str(e):
                    raise Exception(f"API Request Error: {str(e)}")
                else:
                    self.failure_count += 1
                    if self.failure_count >= 3:
                        self.circuit_open = True
                    if attempt == max_retries - 1:
                        raise Exception(f"API Error after {max_retries} attempts: {str(e)}")

rate_limiter = GeminiRateLimiter()

def extract_text_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(['script', 'style', 'nav', 'header', 'footer']):
            script.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main'])
        if main_content:
            text = ' '.join(main_content.stripped_strings)
        else:
            text = ' '.join(soup.stripped_strings)
        return text[:10000]
    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {str(e)}")
        st.error(f"Error extracting text from URL: {str(e)}")
        return ""

def search_entities(query: str, df: pd.DataFrame) -> pd.DataFrame:
    if not query or df.empty:
        return df
    query = query.lower().strip()
    query_parts = query.split()
    def search_row(row):
        try:
            if query in row['entity_name'].lower():
                return True
            if pd.notna(row['attributes']) and any(part in row['attributes'].lower() for part in query_parts):
                return True
            if pd.notna(row['relationships']) and any(part in row['relationships'].lower() for part in query_parts):
                return True
            if 'detailed_attributes' in row and pd.notna(row['detailed_attributes']):
                try:
                    attrs = json.loads(row['detailed_attributes'])
                    for value in attrs.values():
                        if isinstance(value, str) and any(part in value.lower() for part in query_parts):
                            return True
                except:
                    pass
            return False
        except Exception:
            return False
    try:
        mask = df.apply(search_row, axis=1)
        return df[mask]
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return df

def json_serial(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f'Type {type(obj)} not serializable')

def extract_relationship_type(relationship_text: str) -> str:
    if not relationship_text:
        return "related to"
    relationship_lower = relationship_text.lower()
    healthcare_relationships = {
        'works at': 'employee','employed by': 'employee','affiliated with': 'affiliation',
        'practices at': 'practices_at','consultant at': 'consultant','head of': 'department_head',
        'specializes in': 'specialization','located in': 'location','has pharmacy': 'pharmacy_service',
        'has ambulance': 'ambulance_service','available at': 'availability'
    }
    for pattern, rel_type in healthcare_relationships.items():
        if pattern in relationship_lower:
            return rel_type
    if " as " in relationship_lower:
        parts = relationship_lower.split(" as ")
        if len(parts) > 1:
            return parts[-1].strip()
    return "related to"

def extract_target_entity(relationship_text: str) -> str:
    if not relationship_text:
        return ""
    relationship_lower = relationship_text.lower()
    patterns = [
        (r'connected to (.*?)( as | in | via | through |$)', 1),(r'related to (.*?)( as | in | via | through |$)', 1),
        (r'works at (.*?)( as | in | via | through |$)', 1),(r'employed by (.*?)( as | in | via | through |$)', 1),
        (r'affiliated with (.*?)( as | in | via | through |$)', 1),(r'practices at (.*?)( as | in | via | through |$)', 1),
        (r'located in (.*?)( as | in | via | through |$)', 1),(r'specializes in (.*?)( as | in | via | through |$)', 1)
    ]
    for pattern, group_idx in patterns:
        match = re.search(pattern, relationship_lower)
        if match:
            target = match.group(group_idx).strip()
            target = re.sub(r'\s*(as|in|via|through)\s*$', '', target).strip()
            return target
    return ""

def normalize_entity_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip().lower())

def parse_ai_response(response_text: str) -> List[Dict]:
    entities_data = []
    current_entity = None
    current_data = {}
    section_headers = [
        "extracted healthcare entities","extracted entities","entities details and relationships",
        "entities and relationships","entity details","identified entities","entity relationships"
    ]
    lines = response_text.replace('\r\n', '\n').split('\n')
    entity_section_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower() in section_headers:
            entity_section_started = True
            continue
        if not entity_section_started:
            continue
        if not line.startswith(('•', '-', '*', '◦', '‣')) and not re.match(r'^\d+\.', line):
            if current_data:
                entities_data.append(current_data)
            current_data = {'entity_name': line,'attributes': [],'relationships': [],'entity_type': None,'detailed_attributes': {}}
            current_entity = line
        else:
            if not current_entity:
                continue
            detail = re.sub(r'^[•\-\*\d\.◦‣\s]+', '', line).strip()
            relationship_indicators = [
                'connected to', 'related to', 'linked to', 'associated with','works at', 'employed by', 'affiliated with', 'practices at',
                'located in', 'specializes in', 'has pharmacy', 'has ambulance','available at', 'consultant at', 'head of'
            ]
            is_relationship = any(indicator in detail.lower() for indicator in relationship_indicators)
            if is_relationship:
                current_data['relationships'].append(detail)
            else:
                if ':' in detail:
                    attr_parts = detail.split(':', 1)
                    attr_name = attr_parts[0].strip().lower()
                    attr_value = attr_parts[1].strip()
                    if attr_name in ['type', 'category', 'kind']:
                        current_data['entity_type'] = attr_value.lower()
                    else:
                        current_data['attributes'].append(f"{attr_name}: {attr_value}")
                        current_data['detailed_attributes'][attr_name] = attr_value
                else:
                    current_data['attributes'].append(detail)
    if current_data:
        entities_data.append(current_data)
    processed_data = []
    for idx, data in enumerate(entities_data, 1):
        if not data.get('entity_name'):
            continue
        if not data.get('entity_type'):
            data['entity_type'] = detect_entity_type(data['entity_name'],data['attributes'])
        relationship_details = []
        associated_entities = set()
        for rel in data['relationships']:
            target_entity = extract_target_entity(rel)
            if target_entity:
                associated_entities.add(target_entity)
                relationship_type = extract_relationship_type(rel)
                relationship_details.append({
                    'target_entity': target_entity,'relationship_type': relationship_type,'full_description': rel,
                    'source_entity': data['entity_name'],'source_type': data.get('entity_type', 'general')
                })
        clean_attributes = []
        for attr in data['attributes']:
            if attr.lower().startswith('attribute:'):
                attr = attr[len('attribute:'):].strip()
            clean_attributes.append(attr)
        processed_data.append({
            'entity_id': idx,'entity_name': data['entity_name'],'associated_entities': ', '.join(sorted(associated_entities)) if associated_entities else '',
            'attributes': '; '.join(clean_attributes) if clean_attributes else '','relationships': '; '.join(data['relationships']) if data['relationships'] else '',
            'relationship_details': relationship_details,'weight': 1.0,'entity_type': data['entity_type'],'detailed_attributes': data.get('detailed_attributes', {})
        })
    return processed_data

def init_database():
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        cursor = conn.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('''CREATE TABLE IF NOT EXISTS entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,entity_name TEXT NOT NULL,associated_entities TEXT,attributes TEXT,
            relationships TEXT,relationship_details TEXT,detailed_attributes TEXT,weight FLOAT DEFAULT 1.0,entity_type TEXT DEFAULT 'general',
            frequency INTEGER DEFAULT 1,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT,source_identifier TEXT,is_hidden BOOLEAN DEFAULT 0,UNIQUE(entity_name, source_type, source_identifier))''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(entity_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_weight ON entities(weight)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON entities(created_at)')
        cursor.execute('''CREATE TABLE IF NOT EXISTS relationship_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,source_id INTEGER,target_id INTEGER,relationship_type TEXT,
            full_description TEXT,source_entity TEXT,source_entity_type TEXT,weight FLOAT DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,is_hidden BOOLEAN DEFAULT 0,UNIQUE(source_id, target_id, relationship_type),
            FOREIGN KEY(source_id) REFERENCES entities(entity_id) ON DELETE CASCADE,FOREIGN KEY(target_id) REFERENCES entities(entity_id) ON DELETE CASCADE)''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_id ON relationship_edges(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target_id ON relationship_edges(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationship_type ON relationship_edges(relationship_type)')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.error(f"Database initialization error: {e}")

def calculate_entity_weights():
    try:
        conn = sqlite3.connect('entities.db')
        cursor = conn.cursor()
        cursor.execute('''SELECT e.entity_id,COUNT(DISTINCT r1.edge_id) + COUNT(DISTINCT r2.edge_id) as connection_count,
            COUNT(DISTINCT r1.source_id) as out_connections,COUNT(DISTINCT r2.target_id) as in_connections,e.frequency,
            LENGTH(e.attributes) - LENGTH(REPLACE(e.attributes, ';', '')) + 1 as attribute_count,e.entity_type,
            json_array_length(e.detailed_attributes) as detailed_attr_count FROM entities e
            LEFT JOIN relationship_edges r1 ON e.entity_id = r1.source_id AND r1.is_hidden = 0
            LEFT JOIN relationship_edges r2 ON e.entity_id = r2.target_id AND r2.is_hidden = 0
            WHERE e.is_hidden = 0 GROUP BY e.entity_id''')
        weights_data = cursor.fetchall()
        entity_weights = {}
        if not weights_data:
            return
        max_connections = 1
        max_attributes = 1
        max_frequency = 1
        max_detailed_attrs = 1
        for row in weights_data:
            entity_id, connection_count, out_conn, in_conn, frequency, attribute_count, entity_type, detailed_attr_count = row
            max_connections = max(max_connections, connection_count)
            max_attributes = max(max_attributes, attribute_count if attribute_count else 0)
            max_frequency = max(max_frequency, frequency if frequency else 0)
            max_detailed_attrs = max(max_detailed_attrs, detailed_attr_count if detailed_attr_count else 0)
        for row in weights_data:
            entity_id, connection_count, out_conn, in_conn, frequency, attribute_count, entity_type, detailed_attr_count = row
            norm_connections = connection_count / max_connections if max_connections > 0 else 0
            norm_attributes = (attribute_count / max_attributes) if max_attributes > 0 and attribute_count else 0
            norm_frequency = (frequency / max_frequency) if max_frequency > 0 and frequency else 0
            norm_detailed_attrs = (detailed_attr_count / max_detailed_attrs) if max_detailed_attrs > 0 and detailed_attr_count else 0
            type_multiplier = 1.0
            if entity_type:
                if entity_type.lower() in ['hospital', 'clinic']:
                    type_multiplier = 2.0
                elif entity_type.lower() in ['doctor', 'physician']:
                    type_multiplier = 1.8
                elif entity_type.lower() in ['organization', 'medical_center']:
                    type_multiplier = 1.5
            raw_weight = ((norm_connections * 0.4) + (norm_detailed_attrs * 0.3) + (norm_attributes * 0.1) + (norm_frequency * 0.2)) * type_multiplier
            final_weight = 1 + (raw_weight * 9)
            entity_weights[entity_id] = final_weight
        for entity_id, weight in entity_weights.items():
            cursor.execute('UPDATE entities SET weight = ? WHERE entity_id = ?', (weight, entity_id))
        cursor.execute('''SELECT r.edge_id, r.source_id, r.target_id, e1.weight as source_weight, e2.weight as target_weight
            FROM relationship_edges r JOIN entities e1 ON r.source_id = e1.entity_id AND e1.is_hidden = 0
            JOIN entities e2 ON r.target_id = e2.entity_id AND e2.is_hidden = 0 WHERE r.is_hidden = 0''')
        edge_data = cursor.fetchall()
        for row in edge_data:
            edge_id, source_id, target_id, source_weight, target_weight = row
            edge_weight = (source_weight + target_weight) / 2
            cursor.execute('UPDATE relationship_edges SET weight = ? WHERE edge_id = ?', (edge_weight, edge_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error calculating entity weights: {e}")

def detect_entity_type(entity_name: str, attributes: List[str]) -> str:
    entity_name_lower = entity_name.lower()
    attributes_text = " ".join(attributes).lower()
    for attr in attributes:
        if ':' in attr:
            attr_name, attr_value = attr.split(':', 1)
            if attr_name.strip().lower() in ['type', 'category', 'kind']:
                attr_value = attr_value.strip().lower()
                if attr_value in ['hospital', 'clinic', 'doctor', 'physician', 'medical_center', 'pharmacy']:
                    return attr_value
    if any(word in entity_name_lower for word in ['hospital', 'medical center', 'health center']):
        return "hospital"
    elif 'clinic' in entity_name_lower:
        return "clinic"
    elif any(word in entity_name_lower for word in ['dr.', 'doctor', 'physician', 'surgeon', 'specialist']):
        return "doctor"
    elif 'pharmacy' in entity_name_lower:
        return "pharmacy"
    elif 'ambulance' in entity_name_lower:
        return "ambulance_service"
    healthcare_indicators = ['medical', 'healthcare', 'patient', 'treatment', 'surgery','emergency', 'ward', 'department', 'specialization', 'consultation']
    for indicator in healthcare_indicators:
        if indicator in attributes_text:
            if any(word in entity_name_lower for word in ['dr', 'doctor']):
                return "doctor"
            else:
                return "hospital"
    person_indicators = ['mr.', 'mrs.', 'ms.', 'prof.']
    for indicator in person_indicators:
        if indicator in entity_name_lower:
            return "person"
    org_indicators = ['company', 'corporation', 'inc', 'llc', 'ltd']
    for indicator in org_indicators:
        if indicator in entity_name_lower:
            return "organization"
    loc_indicators = ['street', 'avenue', 'road', 'city', 'state']
    for indicator in loc_indicators:
        if indicator in entity_name_lower:
            return "location"
    return "general"

def store_in_database(data: List[Dict], source_type: str, source_identifier: str) -> None:
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        cursor = conn.cursor()
        entity_name_to_id = {}
        for entry in data:
            entry['entity_name'] = entry['entity_name'].strip()
            if not entry['entity_name']:
                continue
            normalized_name = normalize_entity_name(entry['entity_name'])
            cursor.execute('SELECT entity_id, entity_name FROM entities WHERE LOWER(entity_name) = LOWER(?) AND source_type = ? AND source_identifier = ? LIMIT 1',(entry['entity_name'], source_type, source_identifier))
            existing_entity = cursor.fetchone()
            if existing_entity:
                entity_id, existing_name = existing_entity
                cursor.execute('SELECT * FROM entities WHERE entity_id = ?', (entity_id,))
                existing_row = cursor.fetchone()
                existing_data = {
                    'entity_id': existing_row[0],'entity_name': existing_row[1],'associated_entities': existing_row[2] or "",
                    'attributes': existing_row[3] or "",'relationships': existing_row[4] or "",'relationship_details': existing_row[5] or "[]",
                    'detailed_attributes': existing_row[6] or "{}",'weight': existing_row[7] or 1.0,'entity_type': existing_row[8] or "general",'frequency': existing_row[9] or 1
                }
                merged_data = merge_entity_details(existing_data, entry)
                cursor.execute('''UPDATE entities SET associated_entities = ?, attributes = ?, relationships = ?,
                    relationship_details = ?, detailed_attributes = ?, weight = ?, entity_type = ?, frequency = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE entity_id = ?''',(merged_data['associated_entities'], merged_data['attributes'], merged_data['relationships'],merged_data['relationship_details'],
                     merged_data.get('detailed_attributes', '{}'), merged_data.get('weight', 1.0), merged_data['entity_type'], merged_data['frequency'], entity_id))
                entity_name_to_id[normalized_name] = entity_id
            else:
                entry['relationship_details'] = json.dumps(entry.get('relationship_details', []), default=json_serial)
                entry['detailed_attributes'] = json.dumps(entry.get('detailed_attributes', {}), default=json_serial)
                cursor.execute('''INSERT INTO entities (entity_name, associated_entities, attributes, relationships, 
                    relationship_details, detailed_attributes, weight, entity_type, frequency, source_type, source_identifier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',(entry['entity_name'], entry['associated_entities'], entry['attributes'], entry['relationships'], entry['relationship_details'],
                     entry['detailed_attributes'], entry.get('weight', 1.0), entry.get('entity_type', 'general'), 1, source_type, source_identifier))
                entity_id = cursor.lastrowid
                entity_name_to_id[normalized_name] = entity_id
        for entry in data:
            normalized_name = normalize_entity_name(entry['entity_name'])
            source_id = entity_name_to_id.get(normalized_name)
            if not source_id:
                continue
            if 'relationship_details' in entry and isinstance(entry['relationship_details'], str):
                try:
                    relationship_details = json.loads(entry['relationship_details'])
                except json.JSONDecodeError:
                    relationship_details = []
            else:
                relationship_details = entry.get('relationship_details', [])
            for rel in relationship_details:
                if not isinstance(rel, dict):
                    continue
                target_entity_name = rel.get('target_entity', '').strip()
                if not target_entity_name:
                    continue
                normalized_target = normalize_entity_name(target_entity_name)
                target_id = entity_name_to_id.get(normalized_target)
                if not target_id:
                    cursor.execute('SELECT entity_id FROM entities WHERE LOWER(entity_name) = LOWER(?)',(target_entity_name,))
                    existing_target = cursor.fetchone()
                    if existing_target:
                        target_id = existing_target[0]
                        entity_name_to_id[normalized_target] = target_id
                    else:
                        target_type = detect_entity_type(target_entity_name, [])
                        cursor.execute('INSERT INTO entities (entity_name, entity_type, source_type, source_identifier) VALUES (?, ?, ?, ?)',(target_entity_name, target_type, source_type, source_identifier))
                        target_id = cursor.lastrowid
                        entity_name_to_id[normalized_target] = target_id
                relationship_type = rel.get('relationship_type', 'related to')
                full_description = rel.get('full_description',f"{entry['entity_name']} is related to {target_entity_name} as {relationship_type}")
                cursor.execute('SELECT edge_id FROM relationship_edges WHERE source_id = ? AND target_id = ? AND relationship_type = ?',(source_id, target_id, relationship_type))
                existing_edge = cursor.fetchone()
                if not existing_edge:
                    cursor.execute('''INSERT INTO relationship_edges (source_id, target_id, relationship_type, full_description, 
                        source_entity, source_entity_type, weight) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (source_id, target_id, relationship_type, full_description, entry['entity_name'], entry.get('entity_type', 'general'), 1.0))
        conn.commit()
        conn.close()
        calculate_entity_weights()
    except Exception as e:
        logger.error(f"Error storing in database: {e}")
        st.error(f"Database storage error: {e}")

def merge_entity_details(existing: Dict, new: Dict) -> Dict:
    merged = existing.copy()
    existing_associates = set(existing['associated_entities'].split(', ')) if existing['associated_entities'] else set()
    new_associates = set(new['associated_entities'].split(', ')) if new['associated_entities'] else set()
    merged['associated_entities'] = ', '.join(sorted(existing_associates | new_associates - {''}))
    existing_attrs = set(existing['attributes'].split('; ')) if existing['attributes'] else set()
    new_attrs = set(new['attributes'].split('; ')) if new['attributes'] else set()
    merged['attributes'] = '; '.join(sorted(existing_attrs | new_attrs - {''}))
    existing_rels = set(existing['relationships'].split('; ')) if existing['relationships'] else set()
    new_rels = set(new['relationships'].split('; ')) if new['relationships'] else set()
    merged['relationships'] = '; '.join(sorted(existing_rels | new_rels - {''}))
    existing_rel_details = []
    if existing['relationship_details']:
        try:
            existing_rel_details = json.loads(existing['relationship_details'])
        except json.JSONDecodeError:
            pass
    new_rel_details = new.get('relationship_details', [])
    existing_rel_lookup = {}
    for detail in existing_rel_details:
        if isinstance(detail, dict) and 'target_entity' in detail and 'relationship_type' in detail:
            key = (detail['target_entity'].lower(), detail['relationship_type'].lower())
            existing_rel_lookup[key] = detail
    for detail in new_rel_details:
        if isinstance(detail, dict) and 'target_entity' in detail and 'relationship_type' in detail:
            key = (detail['target_entity'].lower(), detail['relationship_type'].lower())
            if key not in existing_rel_lookup:
                existing_rel_details.append(detail)
    merged['relationship_details'] = json.dumps(existing_rel_details, default=json_serial)
    existing_detailed_attrs = {}
    if existing.get('detailed_attributes'):
        try:
            existing_detailed_attrs = json.loads(existing['detailed_attributes'])
        except (json.JSONDecodeError, TypeError):
            existing_detailed_attrs = {}
    new_detailed_attrs = new.get('detailed_attributes', {})
    if isinstance(new_detailed_attrs, str):
        try:
            new_detailed_attrs = json.loads(new_detailed_attrs)
        except json.JSONDecodeError:
            new_detailed_attrs = {}
    merged_detailed_attrs = {**existing_detailed_attrs, **new_detailed_attrs}
    merged['detailed_attributes'] = json.dumps(merged_detailed_attrs, default=json_serial)
    existing_freq = int(existing.get('frequency', 1)) if str(existing.get('frequency', 1)).isdigit() else 1
    merged['frequency'] = existing_freq + 1
    if new.get('entity_type') and new['entity_type'].lower() != 'general':
        merged['entity_type'] = new['entity_type']
    return merged

def process_input_with_model(text: str, source_type: str, source_identifier: str, model_name: str = "gemini-2.5-flash"):
    prompt = """Extract ONLY healthcare-related entities from the following text with focus on hospitals, clinics, and doctors. Extract the following information:

For Hospitals and Clinics:
- Name of hospital/clinic
- Location (address, city, etc.)
- Service hours/opening hours
- Pharmacy services (yes/no/availability)
- Ambulance services (yes/no/availability)

For Doctors:
- Name of doctor
- Specialization/medical field
- Years of experience
- Success rates (if mentioned)
- Consultation fees
- Availability (schedule, days, hours)
- Associated hospital/clinic

Format your response exactly as follows:

Extracted Healthcare Entities
• [Hospital/Clinic Name]
• [Doctor Name]

Entities Details and Relationships
[Hospital/Clinic Name]
• Type: Hospital/Clinic
• Location: [Address, city, area]
• Service Hours: [Opening hours]
• Pharmacy: [Yes/No/Details]
• Ambulance: [Yes/No/Details]
• Connected to [Doctor Name] as [employed doctor/affiliated doctor]

[Doctor Name]
• Type: Doctor
• Specialization: [Medical specialization]
• Experience: [Years of experience]
• Success Rate: [If available]
• Consultation Fees: [Fee information]
• Availability: [Schedule information]
• Connected to [Hospital/Clinic Name] as [works at/affiliated with]

Important guidelines:
1. Extract ONLY hospitals, clinics, and doctors - ignore other entities
2. Focus on location, service hours, pharmacy, ambulance for hospitals/clinics
3. Focus on specialization, experience, success rates, fees, availability for doctors
4. Clearly specify relationships between doctors and hospitals/clinics
5. If information is not available, mark as "Not specified"
6. Be precise with medical specializations and service details

Text: """ + text[:8000]
    try:
        model = genai.GenerativeModel(model_name)
        response = rate_limiter.make_request(model, prompt)
        if response and response.text:
            entities_data = parse_ai_response(response.text)
            if entities_data:
                store_in_database(entities_data, source_type, source_identifier)
                display_data = []
                for item in entities_data:
                    display_item = {
                        'entity_name': item['entity_name'],'entity_type': item['entity_type'],'attributes': item['attributes'],
                        'relationships': item['relationships'],'associated_entities': item['associated_entities'],'weight': item['weight']
                    }
                    display_data.append(display_item)
                display_df = pd.DataFrame(display_data)
                st.success(f"✅ Successfully extracted {len(entities_data)} healthcare entities!")
                st.subheader("📋 Extracted Healthcare Information")
                def color_entity_type(val):
                    color_map = {
                        'hospital': '#1f77b4','clinic': '#1f77b4','doctor': '#ff7f0e','pharmacy': '#2ca02c',
                        'ambulance_service': '#d62728','person': '#9467bd','organization': '#8c564b','location': '#e377c2','general': '#7f7f7f'
                    }
                    color = color_map.get(val.lower(), '#7f7f7f')
                    return f'background-color: {color}; color: white;'
                try:
                    styled_df = display_df.style.map(color_entity_type, subset=['entity_type'])
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                except Exception as e:
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                csv = display_df.to_csv(index=False)
                st.download_button(label="📥 Download CSV",data=csv,file_name=f"entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",mime="text/csv",key=f"download_csv_{uuid.uuid4()}")
                return True
    except Exception as e:
        logger.error(f"Error in process_input_with_model: {e}")
        st.error(f"❌ Error processing input: {str(e)}")
        if "400" in str(e):
            st.info("💡 API model error. Trying fallback model...")
            try:
                return process_input_with_model(text, source_type, source_identifier, "gemini-2.5-flash")
            except:
                st.error("Please check your API key and model availability")
        elif "429" in str(e):
            st.info("💡 Rate limit exceeded. Please wait a minute and try again.")
        elif "401" in str(e):
            st.error("🔑 Invalid API key. Please check your Gemini API key configuration.")
    return False

def process_input(text: str, source_type: str, source_identifier: str):
    return process_input_with_model(text, source_type, source_identifier, "gemini-2.5-flash")

def get_stored_entities() -> pd.DataFrame:
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        df = pd.read_sql_query('''SELECT entity_id, entity_name, associated_entities, attributes, 
            relationships, relationship_details, detailed_attributes, weight, entity_type, frequency, created_at, 
            updated_at, source_type, source_identifier FROM entities WHERE is_hidden = 0
            ORDER BY weight DESC, updated_at DESC''', conn)
        if not df.empty and 'entity_id' in df.columns:
            df['entity_id'] = df['entity_id'].astype(int)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error retrieving stored entities: {e}")
        st.error(f"Error retrieving data: {e}")
        return pd.DataFrame()

def get_relationship_edges() -> pd.DataFrame:
    try:
        conn = sqlite3.connect('entities.db', timeout=30)
        df = pd.read_sql_query('''SELECT e.edge_id, e.source_id, s.entity_name as source_name, 
            e.target_id, t.entity_name as target_name, e.relationship_type, e.full_description, e.weight,
            e.source_entity, e.source_entity_type FROM relationship_edges e 
            JOIN entities s ON e.source_id = s.entity_id JOIN entities t ON e.target_id = t.entity_id
            WHERE e.is_hidden = 0 AND s.is_hidden = 0 AND t.is_hidden = 0 ORDER BY e.weight DESC''', conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error retrieving relationship edges: {e}")
        return pd.DataFrame()

def find_all_paths(edges_df: pd.DataFrame, start_id: int, end_id: int, max_depth: int = 4) -> List[List[int]]:
    graph = {}
    for _, edge in edges_df.iterrows():
        source_id = edge['source_id']
        target_id = edge['target_id']
        if source_id not in graph:
            graph[source_id] = []
        graph[source_id].append(target_id)
    paths = []
    def dfs(current_id, current_path, visited):
        if len(current_path) > max_depth:
            return
        if current_id == end_id:
            paths.append(current_path.copy())
            return
        if current_id not in graph:
            return
        for next_id in graph[current_id]:
            if next_id not in visited:
                visited.add(next_id)
                current_path.append(next_id)
                dfs(next_id, current_path, visited)
                current_path.pop()
                visited.remove(next_id)
    visited = {start_id}
    dfs(start_id, [start_id], visited)
    return paths

def get_entity_connections(entity_id: int, edges_df: pd.DataFrame) -> Dict:
    connections = {'outgoing': [],'incoming': []}
    for _, edge in edges_df.iterrows():
        if edge['source_id'] == entity_id:
            connections['outgoing'].append({
                'target_id': edge['target_id'],'target_name': edge['target_name'],'relationship_type': edge['relationship_type'],
                'full_description': edge['full_description'],'weight': edge['weight'],'source_entity_type': edge.get('source_entity_type', 'general')
            })
        elif edge['target_id'] == entity_id:
            connections['incoming'].append({
                'source_id': edge['source_id'],'source_name': edge['source_name'],'relationship_type': edge['relationship_type'],
                'full_description': edge['full_description'],'weight': edge['weight'],'source_entity_type': edge.get('source_entity_type', 'general')
            })
    connections['outgoing'] = sorted(connections['outgoing'], key=lambda x: x['weight'], reverse=True)
    connections['incoming'] = sorted(connections['incoming'], key=lambda x: x['weight'], reverse=True)
    return connections

def initialize_graph_state():
    if 'graph_nodes' not in st.session_state:
        st.session_state.graph_nodes = set()
    if 'graph_edges' not in st.session_state:
        st.session_state.graph_edges = set()
    if 'expanded_nodes' not in st.session_state:
        st.session_state.expanded_nodes = set()
    if 'node_colors' not in st.session_state:
        st.session_state.node_colors = {}
    if 'entity_types' not in st.session_state:
        st.session_state.entity_types = {}
    if 'hierarchy_level' not in st.session_state:
        st.session_state.hierarchy_level = {}
    if 'node_parents' not in st.session_state:
        st.session_state.node_parents = {}
    if 'expanded_by_type' not in st.session_state:
        st.session_state.expanded_by_type = {}
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    if 'relationship_filter' not in st.session_state:
        st.session_state.relationship_filter = None
    if 'graph_options' not in st.session_state:
        st.session_state.graph_options = {
            'node_size_multiplier': 5,'edge_width_multiplier': 0.5,'color_scheme': 'default','physics_enabled': True,
            'hierarchical_layout': False,'cluster_nodes': True,'show_edge_labels': True,'show_node_labels': True,
            'show_attribute_details': False,'expansion_limit': 5,'dark_mode': False,'node_shape': 'dot',
            'edge_smoothness': 'continuous','font_size': 12
        }
    if 'node_visibility' not in st.session_state:
        st.session_state.node_visibility = {}
    if 'edge_visibility' not in st.session_state:
        st.session_state.edge_visibility = {}
    if 'targeted_expansion' not in st.session_state:
        st.session_state.targeted_expansion = {}
    if 'continuous_expansion' not in st.session_state:
        st.session_state.continuous_expansion = {}

def get_entity_type_color(entity_type):
    color_map = {
        'hospital': '#1f77b4','clinic': '#1f77b4','doctor': '#ff7f0e','pharmacy': '#2ca02c',
        'ambulance_service': '#d62728','person': '#9467bd','organization': '#8c564b','location': '#e377c2','general': '#7f7f7f'
    }
    return color_map.get(entity_type.lower(), '#7f7f7f')

def get_related_entities_by_type(entity_id, entity_type, connection_direction="both"):
    edges_df = get_relationship_edges()
    stored_df = get_stored_entities()
    related_entities = []
    if connection_direction in ["outgoing", "both"]:
        outgoing = edges_df[edges_df['source_id'] == entity_id]
        for _, edge in outgoing.iterrows():
            target_id = edge['target_id']
            target_data = stored_df[stored_df['entity_id'] == target_id]
            if not target_data.empty and target_data.iloc[0]['entity_type'].lower() == entity_type.lower():
                related_entities.append({
                    'entity_id': target_id,'entity_name': edge['target_name'],'entity_type': target_data.iloc[0]['entity_type'],
                    'weight': target_data.iloc[0]['weight'],'relationship': edge['relationship_type'],'direction': 'outgoing'
                })
    if connection_direction in ["incoming", "both"]:
        incoming = edges_df[edges_df['target_id'] == entity_id]
        for _, edge in incoming.iterrows():
            source_id = edge['source_id']
            source_data = stored_df[stored_df['entity_id'] == source_id]
            if not source_data.empty and source_data.iloc[0]['entity_type'].lower() == entity_type.lower():
                related_entities.append({
                    'entity_id': source_id,'entity_name': edge['source_name'],'entity_type': source_data.iloc[0]['entity_type'],
                    'weight': source_data.iloc[0]['weight'],'relationship': edge['relationship_type'],'direction': 'incoming'
                })
    return related_entities

def get_entity_types_from_connections(entity_id):
    edges_df = get_relationship_edges()
    stored_df = get_stored_entities()
    entity_types = set()
    outgoing = edges_df[edges_df['source_id'] == entity_id]
    for _, edge in outgoing.iterrows():
        target_id = edge['target_id']
        target_data = stored_df[stored_df['entity_id'] == target_id]
        if not target_data.empty:
            entity_types.add(target_data.iloc[0]['entity_type'])
    incoming = edges_df[edges_df['target_id'] == entity_id]
    for _, edge in incoming.iterrows():
        source_id = edge['source_id']
        source_data = stored_df[stored_df['entity_id'] == source_id]
        if not source_data.empty:
            entity_types.add(source_data.iloc[0]['entity_type'])
    return list(entity_types)

def expand_node_by_type(entity_id, entity_type):
    if entity_id not in st.session_state.expanded_by_type:
        st.session_state.expanded_by_type[entity_id] = set()
    if entity_type in st.session_state.expanded_by_type[entity_id]:
        st.session_state.expanded_by_type[entity_id].remove(entity_type)
        return
    st.session_state.expanded_by_type[entity_id].add(entity_type)
    related_entities = get_related_entities_by_type(entity_id, entity_type)
    stored_df = get_stored_entities()
    expansion_limit = st.session_state.graph_options.get('expansion_limit', 5)
    added_count = 0
    for entity in related_entities:
        if added_count >= expansion_limit:
            break
        related_id = entity['entity_id']
        if related_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(related_id)
            related_data = stored_df[stored_df['entity_id'] == related_id]
            if not related_data.empty:
                st.session_state.entity_types[related_id] = related_data.iloc[0]['entity_type']
                st.session_state.node_colors[related_id] = get_entity_type_color(related_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[related_id] = True
        if entity['direction'] == 'outgoing':
            edge = (entity_id, related_id, entity['relationship'])
        else:
            edge = (related_id, entity_id, entity['relationship'])
        st.session_state.graph_edges.add(edge)
        st.session_state.edge_visibility[edge] = True
        if related_id not in st.session_state.node_parents:
            st.session_state.node_parents[related_id] = set()
        st.session_state.node_parents[related_id].add(entity_id)
        added_count += 1

def expand_targeted_entity(entity_id):
    """Expand targeted entity with all its connections and attributes"""
    if entity_id in st.session_state.targeted_expansion:
        return
    
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    
    # Add the targeted entity if not already in graph
    if entity_id not in st.session_state.graph_nodes:
        st.session_state.graph_nodes.add(entity_id)
        entity_data = stored_df[stored_df['entity_id'] == entity_id]
        if not entity_data.empty:
            st.session_state.entity_types[entity_id] = entity_data.iloc[0]['entity_type']
            st.session_state.node_colors[entity_id] = get_entity_type_color(entity_data.iloc[0]['entity_type'])
            st.session_state.node_visibility[entity_id] = True
    
    # Expand all connections for the targeted entity
    connections = get_entity_connections(entity_id, edges_df)
    
    # Add outgoing connections
    for connection in connections['outgoing']:
        target_id = connection['target_id']
        if target_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(target_id)
            target_data = stored_df[stored_df['entity_id'] == target_id]
            if not target_data.empty:
                st.session_state.entity_types[target_id] = target_data.iloc[0]['entity_type']
                st.session_state.node_colors[target_id] = get_entity_type_color(target_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[target_id] = True
        
        edge = (entity_id, target_id, connection['relationship_type'])
        st.session_state.graph_edges.add(edge)
        st.session_state.edge_visibility[edge] = True
        
        if target_id not in st.session_state.node_parents:
            st.session_state.node_parents[target_id] = set()
        st.session_state.node_parents[target_id].add(entity_id)
    
    # Add incoming connections
    for connection in connections['incoming']:
        source_id = connection['source_id']
        if source_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(source_id)
            source_data = stored_df[stored_df['entity_id'] == source_id]
            if not source_data.empty:
                st.session_state.entity_types[source_id] = source_data.iloc[0]['entity_type']
                st.session_state.node_colors[source_id] = get_entity_type_color(source_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[source_id] = True
        
        edge = (source_id, entity_id, connection['relationship_type'])
        st.session_state.graph_edges.add(edge)
        st.session_state.edge_visibility[edge] = True
        
        if source_id not in st.session_state.node_parents:
            st.session_state.node_parents[source_id] = set()
        st.session_state.node_parents[source_id].add(entity_id)
    
    st.session_state.targeted_expansion[entity_id] = True

def continuous_expansion(entity_id, depth=2):
    """Recursively expand entities to create continuous expansion"""
    if depth <= 0:
        return
    
    if entity_id not in st.session_state.continuous_expansion:
        st.session_state.continuous_expansion[entity_id] = depth
    
    # Expand the current entity
    expand_targeted_entity(entity_id)
    
    # Get connections for recursive expansion
    edges_df = get_relationship_edges()
    connections = get_entity_connections(entity_id, edges_df)
    
    # Recursively expand connected entities
    all_connected_ids = set()
    for connection in connections['outgoing']:
        all_connected_ids.add(connection['target_id'])
    for connection in connections['incoming']:
        all_connected_ids.add(connection['source_id'])
    
    for connected_id in all_connected_ids:
        if connected_id not in st.session_state.continuous_expansion:
            continuous_expansion(connected_id, depth - 1)

def generate_enhanced_network_graph():
    """Generate enhanced network graph with continuous expansion support"""
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    net = Network(height="800px",width="100%",bgcolor="#ffffff" if not st.session_state.graph_options['dark_mode'] else "#1a1a1a",
        font_color="black" if not st.session_state.graph_options['dark_mode'] else "white",directed=True,notebook=True,cdn_resources='remote')
    
    physics_options = {
        "enabled": st.session_state.graph_options['physics_enabled'],
        "barnesHut": {"gravitationalConstant": -8000,"centralGravity": 0.3,"springLength": 200,"springConstant": 0.04,"damping": 0.09,"avoidOverlap": 0.1},
        "minVelocity": 0.75,"solver": "barnesHut","stabilization": {"enabled": True,"iterations": 1000,"updateInterval": 25}
    }
    
    if st.session_state.graph_options['hierarchical_layout']:
        physics_options["solver"] = "hierarchicalRepulsion"
        physics_options["hierarchicalRepulsion"] = {"nodeDistance": 150,"centralGravity": 0.0,"springLength": 200,"springConstant": 0.01,"damping": 0.09}
        net.set_options("""var options = {"layout": {"hierarchical": {"enabled": true,"levelSeparation": 150,"nodeSpacing": 100,
                    "treeSpacing": 200,"blockShifting": true,"edgeMinimization": true,"parentCentralization": true,
                    "direction": "UD","sortMethod": "directed"} } }""")
    
    node_options = {
        "borderWidth": 2,"borderWidthSelected": 4,"size": 30,"font": {"size": st.session_state.graph_options['font_size'],"strokeWidth": 2,"align": "center","color": "black" if not st.session_state.graph_options['dark_mode'] else "white"},
        "scaling": {"min": 10,"max": 50},"shadow": {"enabled": True,"color": "rgba(0,0,0,0.5)","size": 10,"x": 5,"y": 5},
        "shapeProperties": {"useBorderWithImage": True},"color": {"border": "#2B7CE9","background": "#97C2FC","highlight": {"border": "#2B7CE9","background": "#D2E5FF"},"hover": {"border": "#2B7CE9","background": "#D2E5FF"}}
    }
    
    edge_options = {
        "arrows": {"to": {"enabled": True,"scaleFactor": 0.5,"type": "arrow"}},"color": {"inherit": True,"highlight": "#ff0000","hover": "#ff0000","opacity": 0.8},
        "font": {"size": st.session_state.graph_options['font_size'] - 2,"strokeWidth": 2 if st.session_state.graph_options['show_edge_labels'] else 0,"align": "middle","color": "black" if not st.session_state.graph_options['dark_mode'] else "white"},
        "smooth": {"type": st.session_state.graph_options['edge_smoothness'],"roundness": 0.15},"selectionWidth": 2,
        "shadow": {"enabled": True,"color": "rgba(0,0,0,0.5)","size": 10,"x": 5,"y": 5},"labelHighlightBold": True
    }
    
    net.options = {"nodes": node_options,"edges": edge_options,"physics": physics_options,"interaction": {
        "hover": True,"multiselect": True,"navigationButtons": True,"keyboard": True,"tooltipDelay": 200,"hideEdgesOnDrag": True,"hideNodesOnDrag": False}}
    
    # Add nodes with enhanced tooltips showing all attributes and relationships
    for node_id in st.session_state.graph_nodes:
        if not st.session_state.node_visibility.get(node_id, True):
            continue
        node_data = stored_df[stored_df['entity_id'] == node_id]
        if not node_data.empty:
            node_row = node_data.iloc[0]
            node_name = node_row['entity_name']
            node_type = node_row['entity_type']
            node_weight = node_row['weight']
            level = 0
            if node_id in st.session_state.node_parents:
                level = max([st.session_state.hierarchy_level.get(p, 0) for p in st.session_state.node_parents[node_id]]) + 1
            st.session_state.hierarchy_level[node_id] = level
            node_color = st.session_state.node_colors.get(node_id, get_entity_type_color(node_type))
            node_size = 15 + (node_weight * st.session_state.graph_options['node_size_multiplier'])
            
            # Enhanced tooltip with all attributes and relationships
            detailed_attrs = ""
            if node_row['detailed_attributes']:
                try:
                    attrs = json.loads(node_row['detailed_attributes'])
                    detailed_attrs = "<br><b>Detailed Attributes:</b><ul>" + "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in attrs.items()]) + "</ul>"
                except:
                    detailed_attrs = ""
            
            # Show all relationships
            connections = []
            for edge in st.session_state.graph_edges:
                if edge[0] == node_id or edge[1] == node_id:
                    if edge[0] == node_id:
                        target_id = edge[1]
                        target_data = stored_df[stored_df['entity_id'] == target_id]
                        if not target_data.empty:
                            connections.append(f"→ {target_data.iloc[0]['entity_name']} ({edge[2]})")
                    else:
                        source_id = edge[0]
                        source_data = stored_df[stored_df['entity_id'] == source_id]
                        if not source_data.empty:
                            connections.append(f"← {source_data.iloc[0]['entity_name']} ({edge[2]})")
            
            # Enhanced tooltip content
            node_tooltip = f"""<div style="max-width: 400px; padding: 10px; background-color: {'#ffffff' if not st.session_state.graph_options['dark_mode'] else '#2d2d2d'}; 
                color: {'black' if not st.session_state.graph_options['dark_mode'] else 'white'}; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.3); border-left: 4px solid {node_color};">
                <h4 style="margin: 0 0 10px 0; padding: 0; color: {node_color}; border-bottom: 1px solid #eee; padding-bottom: 5px;">{node_name}</h4>
                <div style="font-size: 12px; line-height: 1.4;">
                    <p style="margin: 5px 0;"><b>Type:</b> {node_type} | <b>Weight:</b> {node_weight:.2f} | <b>Level:</b> {level}</p>
                    <p style="margin: 5px 0;"><b>Attributes:</b> {len(node_row['attributes'].split(';')) if node_row['attributes'] else 0} | 
                    <b>Connections:</b> {len([e for e in st.session_state.graph_edges if e[0] == node_id or e[1] == node_id])}</p>
                    {detailed_attrs}
                    <br><b>All Connections:</b><ul style="max-height: 200px; overflow-y: auto; margin: 5px 0; padding-left: 15px;">{"".join([f"<li style='margin: 2px 0;'>{conn}</li>" for conn in connections])}</ul>
                </div>
            </div>"""
            
            net.add_node(node_id,label=node_name if st.session_state.graph_options['show_node_labels'] else "",title=node_tooltip,
                color=node_color,size=node_size,level=level,shape=st.session_state.graph_options['node_shape'],borderWidth=2,
                mass=1 + (node_weight * 0.5),group=node_type if st.session_state.graph_options['cluster_nodes'] else None)
    
    # Add edges
    for edge in st.session_state.graph_edges:
        if not st.session_state.edge_visibility.get(edge, True):
            continue
        source_id, target_id, relationship = edge
        edge_data = edges_df[(edges_df['source_id'] == source_id) & (edges_df['target_id'] == target_id)]
        if not edge_data.empty:
            edge_row = edge_data.iloc[0]
            title = edge_row['full_description']
            weight = edge_row['weight']
        else:
            title = relationship
            weight = 1.0
        width = 1 + (weight * st.session_state.graph_options['edge_width_multiplier'])
        net.add_edge(source_id,target_id,title=title,label=relationship if st.session_state.graph_options['show_edge_labels'] else "",
            width=width,smooth=True,arrowStrikethrough=False,hidden=False,selectionWidth=1,color={'inherit': 'both'},font={'strokeWidth': 3})
    
    graph_path = f"temp_graph_{uuid.uuid4()}.html"
    net.save_graph(graph_path)
    
    # Add custom JavaScript for enhanced interactivity
    with open(graph_path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        custom_js = """
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                setTimeout(function() {
                    const container = document.getElementsByClassName("vis-network")[0];
                    network.fit(); 
                    network.stabilize(1000); 
                    network.moveTo({
                        position: {x: 0, y: 0},
                        scale: 0.9,
                        offset: {x: 0, y: 0},
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                    
                    // Enhanced double-click for continuous expansion
                    network.on("doubleClick", function(params) {
                        if (params.nodes.length > 0) {
                            const nodeId = params.nodes[0];
                            const nodeData = network.body.data.nodes.get(nodeId);
                            if (nodeData) {
                                // Trigger continuous expansion
                                console.log("Double-clicked node for expansion:", nodeData.label, nodeId);
                                // This would need to be connected to Streamlit backend
                            }
                        }
                    });
                    
                    network.on("hoverNode", function(params) {
                        network.canvas.body.container.style.cursor = 'pointer';
                    });
                    
                    network.on("blurNode", function(params) {
                        network.canvas.body.container.style.cursor = 'default';
                    });
                    
                }, 500);
            });
        </script>
        """
        content = content.replace('</body>', custom_js + '</body>')
        f.write(content)
        f.truncate()
    
    return graph_path

def toggle_node_expansion(entity_id):
    if entity_id in st.session_state.expanded_nodes:
        st.session_state.expanded_nodes.remove(entity_id)
        nodes_to_remove = set()
        edges_to_remove = set()
        for node_id in st.session_state.graph_nodes:
            if node_id != entity_id and node_id in st.session_state.node_parents:
                if len(st.session_state.node_parents[node_id]) == 1 and entity_id in st.session_state.node_parents[node_id]:
                    nodes_to_remove.add(node_id)
        for edge in st.session_state.graph_edges.copy():
            source_id, target_id, _ = edge
            if source_id in nodes_to_remove or target_id in nodes_to_remove:
                edges_to_remove.add(edge)
        st.session_state.graph_nodes -= nodes_to_remove
        st.session_state.graph_edges -= edges_to_remove
    else:
        st.session_state.expanded_nodes.add(entity_id)
        expand_targeted_entity(entity_id)

def expand_node(entity_id):
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    expansion_limit = st.session_state.graph_options.get('expansion_limit', 5)
    added_count = 0
    outgoing = edges_df[edges_df['source_id'] == entity_id]
    for _, edge in outgoing.iterrows():
        if added_count >= expansion_limit:
            break
        target_id = edge['target_id']
        if target_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(target_id)
            target_data = stored_df[stored_df['entity_id'] == target_id]
            if not target_data.empty:
                st.session_state.entity_types[target_id] = target_data.iloc[0]['entity_type']
                st.session_state.node_colors[target_id] = get_entity_type_color(target_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[target_id] = True
        edge_tuple = (entity_id, target_id, edge['relationship_type'])
        st.session_state.graph_edges.add(edge_tuple)
        st.session_state.edge_visibility[edge_tuple] = True
        if target_id not in st.session_state.node_parents:
            st.session_state.node_parents[target_id] = set()
        st.session_state.node_parents[target_id].add(entity_id)
        added_count += 1
    incoming = edges_df[edges_df['target_id'] == entity_id]
    for _, edge in incoming.iterrows():
        if added_count >= expansion_limit * 2:
            break
        source_id = edge['source_id']
        if source_id not in st.session_state.graph_nodes:
            st.session_state.graph_nodes.add(source_id)
            source_data = stored_df[stored_df['entity_id'] == source_id]
            if not source_data.empty:
                st.session_state.entity_types[source_id] = source_data.iloc[0]['entity_type']
                st.session_state.node_colors[source_id] = get_entity_type_color(source_data.iloc[0]['entity_type'])
                st.session_state.node_visibility[source_id] = True
        edge_tuple = (source_id, entity_id, edge['relationship_type'])
        st.session_state.graph_edges.add(edge_tuple)
        st.session_state.edge_visibility[edge_tuple] = True
        if source_id not in st.session_state.node_parents:
            st.session_state.node_parents[source_id] = set()
        st.session_state.node_parents[source_id].add(entity_id)
        added_count += 1

def reset_graph():
    st.session_state.graph_nodes = set()
    st.session_state.graph_edges = set()
    st.session_state.expanded_nodes = set()
    st.session_state.node_colors = {}
    st.session_state.entity_types = {}
    st.session_state.hierarchy_level = {}
    st.session_state.node_parents = {}
    st.session_state.expanded_by_type = {}
    st.session_state.selected_node = None
    st.session_state.relationship_filter = None
    st.session_state.node_visibility = {}
    st.session_state.edge_visibility = {}
    st.session_state.targeted_expansion = {}
    st.session_state.continuous_expansion = {}

def visualize_knowledge_graph():
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    initialize_graph_state()
    
    st.markdown("""
    <style>
        .graph-section {background-color: #f8f9fa;padding: 1.5rem;border-radius: 8px;margin-bottom: 1.5rem;box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .graph-controls {background-color: white;padding: 1rem;border-radius: 8px;margin-bottom: 1rem;box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
        .entity-list {background-color: white;padding: 1rem;border-radius: 8px;box-shadow: 0 1px 3px rgba(0,0,0,0.1);max-height: 600px;overflow-y: auto;}
        .node-visibility {background-color: white;padding: 1rem;border-radius: 8px;box-shadow: 0 1px 3px rgba(0,0,0,0.1);max-height: 600px;overflow-y: auto;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="graph-section">', unsafe_allow_html=True)
    st.subheader("🌐 Advanced Knowledge Graph Visualization")
    
    with st.expander("⚙️ Graph Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.graph_options['node_size_multiplier'] = st.slider("Node Size Multiplier", 1, 10, 5,help="Adjust the size of nodes based on their importance")
            st.session_state.graph_options['edge_width_multiplier'] = st.slider("Edge Width Multiplier", 0.1, 2.0, 0.5, 0.1,help="Adjust the width of edges based on relationship strength")
            st.session_state.graph_options['font_size'] = st.slider("Font Size", 8, 20, 12,help="Adjust the font size for node and edge labels")
        with col2:
            st.session_state.graph_options['physics_enabled'] = st.checkbox("Enable Physics", True,help="Enable physics simulation for dynamic graph layout")
            st.session_state.graph_options['hierarchical_layout'] = st.checkbox("Hierarchical Layout", False,help="Use hierarchical layout for better visualization of relationships")
            st.session_state.graph_options['cluster_nodes'] = st.checkbox("Cluster by Type", True,help="Group nodes by their entity type")
            st.session_state.graph_options['dark_mode'] = st.checkbox("Dark Mode", False,help="Toggle dark mode for the graph")
        with col3:
            st.session_state.graph_options['show_edge_labels'] = st.checkbox("Show Edge Labels", True,help="Display relationship labels on edges")
            st.session_state.graph_options['show_node_labels'] = st.checkbox("Show Node Labels", True,help="Display entity names on nodes")
            st.session_state.graph_options['show_attribute_details'] = st.checkbox("Show Attribute Details", True,help="Show detailed attributes in node tooltips")
            st.session_state.graph_options['expansion_limit'] = st.slider("Expansion Limit", 1, 10, 5,help="Number of connections to show when expanding a node")
    
    st.markdown('<div class="graph-controls">', unsafe_allow_html=True)
    st.write("### 🔍 Filter Options")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        entity_type_filter = st.multiselect("Filter by Entity Type",["All"] + sorted(stored_df['entity_type'].unique().tolist()),default=["All"],key="entity_type_filter")
    with filter_col2:
        min_weight = st.slider("Minimum Entity Weight",min_value=1.0,max_value=float(stored_df['weight'].max()) if not stored_df.empty else 10.0,value=1.0,step=0.5,key="min_weight")
    with filter_col3:
        search_term = st.text_input("Search Entities",key="graph_search",placeholder="Search by name, attributes, or relationships")
    st.markdown('</div>', unsafe_allow_html=True)
    
    filtered_df = stored_df.copy()
    if "All" not in entity_type_filter:
        filtered_df = filtered_df[filtered_df['entity_type'].isin(entity_type_filter)]
    filtered_df = filtered_df[filtered_df['weight'] >= min_weight]
    if search_term:
        filtered_df = search_entities(search_term, filtered_df)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown('<div class="entity-list">', unsafe_allow_html=True)
        st.write("### 📌 Entities")
        st.write("Click to add/remove entities from the graph:")
        
        if not filtered_df.empty:
            # Continuous Expansion Controls
            st.write("### 🔄 Continuous Expansion")
            expansion_depth = st.slider("Expansion Depth", 1, 5, 2, help="How many levels to expand from selected entity")
            
            entity_types = [et for et in sorted(filtered_df['entity_type'].unique()) 
                          if et.lower() in ['hospital', 'clinic', 'doctor', 'pharmacy', 'ambulance_service', 'person', 'organization', 'location']]
            
            for entity_type in entity_types:
                type_df = filtered_df[filtered_df['entity_type'] == entity_type]
                type_color = get_entity_type_color(entity_type)
                with st.expander(f"📌 {entity_type.capitalize()} ({len(type_df)})", expanded=True):
                    for _, row in type_df.iterrows():
                        entity_id = row['entity_id']
                        entity_name = row['entity_name']
                        entity_weight = row['weight']
                        
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            is_selected = st.session_state.selected_node == entity_id
                            button_color = "primary" if entity_id in st.session_state.graph_nodes else "secondary"
                            
                            if st.button(f"{entity_name} ({entity_weight:.1f})",key=f"entity_{entity_id}",type=button_color,use_container_width=True):
                                if entity_id in st.session_state.graph_nodes:
                                    st.session_state.graph_nodes.remove(entity_id)
                                    st.session_state.graph_edges = {edge for edge in st.session_state.graph_edges if edge[0] != entity_id and edge[1] != entity_id}
                                    if entity_id in st.session_state.expanded_nodes:
                                        st.session_state.expanded_nodes.remove(entity_id)
                                else:
                                    st.session_state.graph_nodes.add(entity_id)
                                    st.session_state.entity_types[entity_id] = entity_type
                                    st.session_state.node_colors[entity_id] = type_color
                                    st.session_state.node_visibility[entity_id] = True
                                st.rerun()
                        
                        with col_b:
                            if st.button("🔍", key=f"expand_{entity_id}", help=f"Expand {entity_name} with all connections"):
                                continuous_expansion(entity_id, expansion_depth)
                                st.rerun()
        
        if st.button("🔄 Reset Graph", type="primary", use_container_width=True):
            reset_graph()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="node-visibility">', unsafe_allow_html=True)
        st.write("### 👁️ Node Visibility")
        if st.session_state.graph_nodes:
            for node_id in sorted(st.session_state.graph_nodes):
                node_data = stored_df[stored_df['entity_id'] == node_id]
                if not node_data.empty:
                    node_name = node_data.iloc[0]['entity_name']
                    current_visibility = st.session_state.node_visibility.get(node_id, True)
                    new_visibility = st.checkbox(f"Show {node_name}",value=current_visibility,key=f"node_vis_{node_id}")
                    if new_visibility != current_visibility:
                        st.session_state.node_visibility[node_id] = new_visibility
                        st.rerun()
        else:
            st.info("No nodes in graph yet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.write("### 🌐 Graph Visualization")
    graph_container = st.container()
    
    if st.session_state.graph_nodes:
        graph_path = generate_enhanced_network_graph()
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_html = f.read()
        graph_html = graph_html.replace('<div id="mynetwork"></div>',
            '<div id="mynetwork" style="width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></div>')
        
        with graph_container:
            components.html(graph_html, height=850, scrolling=False)
        
        try:
            os.remove(graph_path)
        except:
            pass
        
        # Enhanced Node Exploration Section
        st.write("### 🔍 Advanced Node Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Basic Expansion", "Type-Based Expansion", "Continuous Expansion"])
        
        with tab1:
            st.write("Expand individual nodes to see immediate connections:")
            for entity_id in sorted(st.session_state.graph_nodes):
                entity_data = stored_df[stored_df['entity_id'] == entity_id]
                if not entity_data.empty:
                    entity_name = entity_data.iloc[0]['entity_name']
                    entity_type = entity_data.iloc[0]['entity_type']
                    button_color = "primary" if entity_id in st.session_state.expanded_nodes else "secondary"
                    if st.button(f"🔍 Expand {entity_name} ({entity_type})",key=f"expand_basic_{entity_id}",type=button_color,use_container_width=True):
                        toggle_node_expansion(entity_id)
                        st.rerun()
        
        with tab2:
            st.write("Expand nodes by specific connection types:")
            for entity_id in sorted(st.session_state.graph_nodes):
                entity_data = stored_df[stored_df['entity_id'] == entity_id]
                if not entity_data.empty:
                    entity_name = entity_data.iloc[0]['entity_name']
                    connected_types = get_entity_types_from_connections(entity_id)
                    if connected_types:
                        st.write(f"**{entity_name}** connected to:")
                        for entity_type in connected_types:
                            if entity_type.lower() in ['hospital', 'clinic', 'doctor', 'pharmacy', 'ambulance_service', 'person', 'organization', 'location']:
                                if entity_id not in st.session_state.expanded_by_type:
                                    st.session_state.expanded_by_type[entity_id] = set()
                                is_expanded = entity_type in st.session_state.expanded_by_type.get(entity_id, set())
                                button_color = "primary" if is_expanded else "secondary"
                                if st.button(f"🔗 Show {entity_type} connections",key=f"type_{entity_id}_{entity_type}",type=button_color,use_container_width=True):
                                    expand_node_by_type(entity_id, entity_type)
                                    st.rerun()
        
        with tab3:
            st.write("Continuous expansion with configurable depth:")
            expansion_depth_continuous = st.slider("Expansion Depth", 1, 5, 2, key="continuous_depth", help="Number of levels to expand recursively")
            
            for entity_id in sorted(st.session_state.graph_nodes):
                entity_data = stored_df[stored_df['entity_id'] == entity_id]
                if not entity_data.empty:
                    entity_name = entity_data.iloc[0]['entity_name']
                    if st.button(f"🔄 Continuously Expand {entity_name}",key=f"continuous_{entity_id}",type="primary",use_container_width=True):
                        continuous_expansion(entity_id, expansion_depth_continuous)
                        st.rerun()
    
    else:
        st.info("ℹ️ Add entities to the graph from the left sidebar to start visualizing.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def analyze_entity_relationships():
    st.subheader("🔗 Entity Relationship Analysis")
    stored_df = get_stored_entities()
    edges_df = get_relationship_edges()
    
    st.markdown("""
    <style>
        .analysis-section {background-color: #f8f9fa;padding: 1.5rem;border-radius: 8px;margin-bottom: 1.5rem;box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .analysis-controls {background-color: white;padding: 1rem;border-radius: 8px;margin-bottom: 1rem;box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    with st.markdown('<div class="analysis-controls">', unsafe_allow_html=True):
        col1, col2 = st.columns(2)
        with col1:
            entity_type_filter = st.multiselect("Filter by Entity Type",["All"] + sorted(stored_df['entity_type'].unique().tolist()),default=["All"],key="entity_type_filter_analysis")
        with col2:
            min_weight = st.slider("Minimum Entity Weight",min_value=1.0,max_value=float(stored_df['weight'].max()) if not stored_df.empty else 10.0,value=1.0,step=0.5,key="min_weight_analysis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    filtered_df = stored_df.copy()
    if "All" not in entity_type_filter:
        filtered_df = filtered_df[filtered_df['entity_type'].isin(entity_type_filter)]
    filtered_df = filtered_df[filtered_df['weight'] >= min_weight]
    
    with st.markdown('<div class="analysis-controls">', unsafe_allow_html=True):
        col1, col2 = st.columns(2)
        with col1:
            entity1_id = st.selectbox("Select First Entity",options=filtered_df['entity_id'].tolist(),format_func=lambda x: filtered_df[filtered_df['entity_id'] == x].iloc[0]['entity_name'],key="entity1_select")
        with col2:
            entity2_id = st.selectbox("Select Second Entity",options=filtered_df['entity_id'].tolist(),format_func=lambda x: filtered_df[filtered_df['entity_id'] == x].iloc[0]['entity_name'],index=1 if len(filtered_df) > 1 else 0,key="entity2_select")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if entity1_id and entity2_id:
        entity1_name = filtered_df[filtered_df['entity_id'] == entity1_id].iloc[0]['entity_name']
        entity2_name = filtered_df[filtered_df['entity_id'] == entity2_id].iloc[0]['entity_name']
        
        st.write(f"### Analyzing relationship between: **{entity1_name}** and **{entity2_name}**")
        
        direct_relationship = edges_df[(edges_df['source_id'] == entity1_id) & (edges_df['target_id'] == entity2_id)]
        reverse_relationship = edges_df[(edges_df['source_id'] == entity2_id) & (edges_df['target_id'] == entity1_id)]
        
        if not direct_relationship.empty or not reverse_relationship.empty:
            st.write("#### Direct Relationships")
            if not direct_relationship.empty:
                for _, row in direct_relationship.iterrows():
                    st.markdown(f"""<div style="background-color: #e8f4fc; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <b>{entity1_name}</b> → <b>{entity2_name}</b>: {row['full_description']}</div>""", unsafe_allow_html=True)
            if not reverse_relationship.empty:
                for _, row in reverse_relationship.iterrows():
                    st.markdown(f"""<div style="background-color: #e8f4fc; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <b>{entity2_name}</b> → <b>{entity1_name}</b>: {row['full_description']}</div>""", unsafe_allow_html=True)
        
        st.write("#### Connection Paths")
        paths = find_all_paths(edges_df, entity1_id, entity2_id)
        if paths:
            st.write(f"Found {len(paths)} path(s) between these entities:")
            for i, path in enumerate(paths, 1):
                path_str = []
                for j in range(len(path) - 1):
                    source_id = path[j]
                    target_id = path[j + 1]
                    source_name = filtered_df[filtered_df['entity_id'] == source_id].iloc[0]['entity_name']
                    target_name = filtered_df[filtered_df['entity_id'] == target_id].iloc[0]['entity_name']
                    rel_edge = edges_df[(edges_df['source_id'] == source_id) & (edges_df['target_id'] == target_id)]
                    if not rel_edge.empty:
                        relationship = rel_edge.iloc[0]['relationship_type']
                    else:
                        relationship = "related to"
                    path_str.append(f"**{source_name}** → [{relationship}] → **{target_name}**")
                st.markdown(f"""<div style="background-color: #f0f7ff; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <b>Path {i}:</b> {' → '.join(path_str)}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                No connection paths found between <b>{entity1_name}</b> and <b>{entity2_name}</b>.</div>""", unsafe_allow_html=True)
        
        st.write("#### Common Connections")
        entity1_outgoing = set(edges_df[edges_df['source_id'] == entity1_id]['target_id'])
        entity2_outgoing = set(edges_df[edges_df['source_id'] == entity2_id]['target_id'])
        common_outgoing = entity1_outgoing.intersection(entity2_outgoing)
        
        if common_outgoing:
            st.write(f"Both **{entity1_name}** and **{entity2_name}** connect to:")
            for common_id in common_outgoing:
                common_name = filtered_df[filtered_df['entity_id'] == common_id].iloc[0]['entity_name']
                rel1 = edges_df[(edges_df['source_id'] == entity1_id) & (edges_df['target_id'] == common_id)].iloc[0]['relationship_type']
                rel2 = edges_df[(edges_df['source_id'] == entity2_id) & (edges_df['target_id'] == common_id)].iloc[0]['relationship_type']
                st.markdown(f"""<div style="background-color: #e8f5e9; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <b>{common_name}</b> ({entity1_name} {rel1}; {entity2_name} {rel2})</div>""", unsafe_allow_html=True)
        
        entity1_incoming = set(edges_df[edges_df['target_id'] == entity1_id]['source_id'])
        entity2_incoming = set(edges_df[edges_df['target_id'] == entity2_id]['source_id'])
        common_incoming = entity1_incoming.intersection(entity2_incoming)
        
        if common_incoming:
            st.write(f"Both **{entity1_name}** and **{entity2_name}** are connected from:")
            for common_id in common_incoming:
                common_name = filtered_df[filtered_df['entity_id'] == common_id].iloc[0]['entity_name']
                rel1 = edges_df[(edges_df['source_id'] == common_id) & (edges_df['target_id'] == entity1_id)].iloc[0]['relationship_type']
                rel2 = edges_df[(edges_df['source_id'] == common_id) & (edges_df['target_id'] == entity2_id)].iloc[0]['relationship_type']
                st.markdown(f"""<div style="background-color: #e8f5e9; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <b>{common_name}</b> ({common_name} {rel1} {entity1_name}; {common_name} {rel2} {entity2_name})</div>""", unsafe_allow_html=True)
        
        if not common_outgoing and not common_incoming:
            st.markdown(f"""<div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                No common connections found between <b>{entity1_name}</b> and <b>{entity2_name}</b>.</div>""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Healthcare Knowledge Graph Explorer",page_icon="🏥",layout="wide",initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
        .stApp {background-color: #f8f9fa;}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: #2c3e50;}
        .stButton>button {border-radius: 8px;padding: 0.5rem 1rem;font-weight: 500;transition: all 0.2s ease;background-color: #6e8efb;color: white;border: none;}
        .stButton>button:hover {transform: translateY(-2px);box-shadow: 0 2px 6px rgba(110, 142, 251, 0.4);background-color: #5a7de3;}
        .stButton>button:focus {box-shadow: 0 0 0 0.2rem rgba(110, 142, 251, 0.25);}
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {border-radius: 8px;border: 1px solid #ced4da;padding: 0.5rem;}
        .stSelectbox>div>div>select {border-radius: 8px;border: 1px solid #ced4da;padding: 0.5rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 8px;padding: 0 1rem;}
        .stTabs [data-baseweb="tab"] {padding: 0.75rem 1.5rem;border-radius: 8px 8px 0 0;font-weight: 500;transition: all 0.2s ease;background-color: #e9ecef;}
        .stTabs [aria-selected="true"] {background-color: #6e8efb;color: white;}
        .stDataFrame {border-radius: 8px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);border: 1px solid #e0e0e0;}
        .stExpander {border-radius: 8px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);border: 1px solid #e0e0e0;}
        .stExpander .streamlit-expanderHeader {font-weight: 600;color: #2c3e50;}
        ::-webkit-scrollbar {width: 8px;height: 8px;}
        ::-webkit-scrollbar-track {background: #f1f1f1;border-radius: 10px;}
        ::-webkit-scrollbar-thumb {background: #6e8efb;border-radius: 10px;}
        ::-webkit-scrollbar-thumb:hover {background: #5a7de3;}
    </style>
    """, unsafe_allow_html=True)
    
    init_database()
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f77b4, #6e8efb); padding: 2rem; border-radius: 0 0 8px 8px; margin-bottom: 2rem;box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2.5rem;">🏥</div>
            <div>
                <h1 style="color: white; margin: 0;">Healthcare Knowledge Graph Explorer</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 0;">Extract and visualize relationships between hospitals, clinics, and doctors using AI</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("🔧 Configuration")
        model_option = st.selectbox("Select Gemini Model",["gemini-2.5-flash"],index=0,help="Select the Gemini model for entity extraction")
        graph_type = st.selectbox("Graph Visualization Type",["interactive", "plotly", "simple"],index=0,help="Choose how to visualize the knowledge graph")
        st.subheader("📊 System Status")
        try:
            stored_df = get_stored_entities()
            if not stored_df.empty:
                entities = stored_df[stored_df['entity_type'].isin(['hospital', 'clinic', 'doctor'])]
                total_entities = len(stored_df)
                healthcare_count = len(entities)
                st.metric("Total Entities", total_entities)
                st.metric("Healthcare Entities", healthcare_count)
                st.metric("Data Source", "SQLite Database")
            else:
                st.info("No entities stored yet")
        except:
            st.info("Database not initialized")
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                try:
                    conn = sqlite3.connect('entities.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM entities')
                    cursor.execute('DELETE FROM relationship_edges')
                    conn.commit()
                    conn.close()
                    st.success("All data cleared successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {e}")
    
    tab1, tab2, tab3 = st.tabs(["📥 Extract Entities", "🌐 View Knowledge Graph", "🔍 Analyze Relationships"])
    
    with tab1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6e8efb, #a777e3); padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0; display: flex; align-items: center; gap: 10px;">📥 Extract Healthcare Entities</h2>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio("Select Input Method",["Text", "URL", "File"],horizontal=True,key="input_method")
        source_type = "Text"
        source_identifier = "Manual Input"
        
        if input_method == "Text":
            text_input = st.text_area("Enter healthcare-related text:",height=200,key="text_input",
                placeholder="Paste text about hospitals, clinics, doctors, medical services, etc. here...\n\nExample: 'City General Hospital located at 123 Main Street offers 24/7 emergency services with pharmacy and ambulance. Dr. Smith specializes in cardiology with 15 years experience and consultation fees of $200.'")
            if st.button("🚀 Extract Entities", key="extract_text", type="primary"):
                if text_input:
                    with st.spinner("Extracting healthcare entities using AI..."):
                        process_input_with_model(text_input, source_type, source_identifier, model_option)
                else:
                    st.error("Please enter some text to extract entities from.")
        
        elif input_method == "URL":
            url_input = st.text_input("Enter healthcare website URL:",key="url_input",placeholder="https://example-hospital.com/services")
            if st.button("🚀 Extract from URL", key="extract_url", type="primary"):
                if url_input:
                    source_type = "URL"
                    source_identifier = url_input
                    with st.spinner("Extracting text from URL and processing..."):
                        text = extract_text_from_url(url_input)
                        if text:
                            process_input_with_model(text, source_type, source_identifier, model_option)
                        else:
                            st.error("Failed to extract text from the URL.")
                else:
                    st.error("Please enter a valid URL.")
                    
        elif input_method == "File":
            uploaded_file = st.file_uploader("Upload healthcare document", type=['txt', 'pdf', 'docx'])
            if uploaded_file and st.button("🚀 Extract from File", type="primary", use_container_width=True):
                try:
                    if uploaded_file.type == "text/plain":
                        text = str(uploaded_file.read(), "utf-8")
                    else:
                        st.warning("Please upload a text file (.txt) for now")
                        text = ""
                    if text:
                        with st.spinner("Processing document..."):
                            process_input_with_model(text, "file", uploaded_file.name, model_option)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        stored_df = get_stored_entities()
        if not stored_df.empty:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #6e8efb, #a777e3); padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h2 style="color: white; margin: 0; display: flex; align-items: center; gap: 10px;">📚 Stored Entities</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                entity_type_filter = st.multiselect("Filter by Entity Type",["All"] + sorted(stored_df['entity_type'].unique().tolist()),default=["All"],key="entity_type_filter_stored")
            with col2:
                search_query = st.text_input("Search Entities",key="entity_search",placeholder="Search by name, attributes, or relationships")
            
            filtered_df = stored_df.copy()
            if "All" not in entity_type_filter:
                filtered_df = filtered_df[filtered_df['entity_type'].isin(entity_type_filter)]
            if search_query:
                filtered_df = search_entities(search_query, filtered_df)
            
            if not filtered_df.empty:
                def color_entity_type(val):
                    color_map = {
                        'hospital': '#1f77b4','clinic': '#1f77b4','doctor': '#ff7f0e','pharmacy': '#2ca02c',
                        'ambulance_service': '#d62728','person': '#9467bd','organization': '#8c564b','location': '#e377c2','general': '#7f7f7f'
                    }
                    color = color_map.get(val.lower(), '#7f7f7f')
                    return f'background-color: {color}; color: white;'
                
                display_df = filtered_df[['entity_id', 'entity_name', 'entity_type', 'attributes', 'relationships', 'weight']]
                
                try:
                    styled_df = display_df.style.map(color_entity_type, subset=['entity_type'])
                    st.dataframe(styled_df,hide_index=True,use_container_width=True,column_config={
                            "entity_id": st.column_config.NumberColumn("ID"),"entity_name": st.column_config.TextColumn("Name"),
                            "entity_type": st.column_config.TextColumn("Type"),"attributes": st.column_config.TextColumn("Attributes"),
                            "relationships": st.column_config.TextColumn("Relationships"),"weight": st.column_config.NumberColumn("Weight", format="%.2f")
                        })
                except KeyError:
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    with tab2:
        visualize_knowledge_graph()
    
    with tab3:
        analyze_entity_relationships()

if __name__ == "__main__":
    main()
