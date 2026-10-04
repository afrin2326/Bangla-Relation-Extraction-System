# ============================================
# src/utils/file_processor.py
# File Processor with Better Pattern Matching
# ============================================

import os
import re
from typing import List, Dict, Any

class FileProcessor:
    """Process TXT, PDF, DOCX files and extract entities"""
    
    @staticmethod
    def extract_text_from_txt(txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except:
            try:
                with open(txt_path, 'r', encoding='latin1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading TXT: {e}")
                return ""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            print("⚠️ PyPDF2 not installed. PDF support disabled.")
            return ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            import docx
            doc = docx.Document(docx_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        # Bengali sentence boundaries
        sentences = re.split(r'[।\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        return sentences
    
    @staticmethod
    def extract_entities(text: str) -> List[Dict[str, Any]]:
        """
        Extract potential entity pairs from text using multiple patterns
        """
        results = []
        
        # Pattern 1: Person Location-এ জন্মগ্রহণ করেন
        pattern1 = r'([\w\s]+?)\s+([\w\s]+[ঃায়ে])\s+জন্মগ্রহণ\s+করেন'
        matches = re.findall(pattern1, text)
        for match in matches:
            if len(match) >= 2:
                entity1 = match[0].strip()
                entity2 = match[1].strip()
                if entity1 and entity2:
                    results.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'text': f"{entity1} {entity2}"
                    })
        
        # Pattern 2: Person 'Movie' ছবিতে অভিনয় করেছেন
        pattern2 = r'([\w\s]+?)\s+[\'"]([\w\s]+)[\'"]\s+ছবিতে\s+অভিনয়\s+করেছেন'
        matches = re.findall(pattern2, text)
        for match in matches:
            if len(match) >= 2:
                entity1 = match[0].strip()
                entity2 = match[1].strip()
                if entity1 and entity2:
                    results.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'text': f"{entity1} {entity2}"
                    })
        
        # Pattern 3: Person 'Book' লিখেছেন
        pattern3 = r'([\w\s]+?)\s+[\'"]([\w\s]+)[\'"]\s+লিখেছেন'
        matches = re.findall(pattern3, text)
        for match in matches:
            if len(match) >= 2:
                entity1 = match[0].strip()
                entity2 = match[1].strip()
                if entity1 and entity2:
                    results.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'text': f"{entity1} {entity2}"
                    })
        
        # Pattern 4: Person 'Movie' পরিচালনা করেছেন
        pattern4 = r'([\w\s]+?)\s+[\'"]([\w\s]+)[\'"]\s+পরিচালনা\s+করেছেন'
        matches = re.findall(pattern4, text)
        for match in matches:
            if len(match) >= 2:
                entity1 = match[0].strip()
                entity2 = match[1].strip()
                if entity1 and entity2:
                    results.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'text': f"{entity1} {entity2}"
                    })
        
        # Pattern 5: Person Company প্রতিষ্ঠা করেছেন
        pattern5 = r'([\w\s]+?)\s+([\w\s]+)\s+প্রতিষ্ঠা\s+করেছেন'
        matches = re.findall(pattern5, text)
        for match in matches:
            if len(match) >= 2:
                entity1 = match[0].strip()
                entity2 = match[1].strip()
                if entity1 and entity2:
                    results.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'text': f"{entity1} {entity2}"
                    })
        
        # Pattern 6: Person Location-এ জন্ম (simplified)
        pattern6 = r'([\w\s]+?)\s+([\w\s]+[ঃায়ে])\s+জন্ম'
        matches = re.findall(pattern6, text)
        for match in matches:
            if len(match) >= 2:
                entity1 = match[0].strip()
                entity2 = match[1].strip()
                if entity1 and entity2:
                    results.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'text': f"{entity1} {entity2}"
                    })
        
        # Pattern 7: Person 'Book' লিখেছেন (without quotes)
        pattern7 = r'([\w\s]+?)\s+([\w\s]+)\s+লিখেছেন'
        matches = re.findall(pattern7, text)
        for match in matches:
            if len(match) >= 2:
                entity1 = match[0].strip()
                entity2 = match[1].strip()
                # Filter out common words
                if entity1 and entity2 and len(entity1) > 3 and len(entity2) > 2:
                    # Avoid duplicate entries
                    is_duplicate = False
                    for r in results:
                        if r['entity1'] == entity1 and r['entity2'] == entity2:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        results.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'text': f"{entity1} {entity2}"
                        })
        
        return results
    
    @staticmethod
    def process_file(file_path: str) -> Dict[str, Any]:
        """Process any file and return extracted data"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = FileProcessor.extract_text_from_pdf(file_path)
        elif ext == '.txt':
            text = FileProcessor.extract_text_from_txt(file_path)
        elif ext == '.docx':
            text = FileProcessor.extract_text_from_docx(file_path)
        else:
            return {'success': False, 'error': f'Unsupported file: {ext}'}
        
        if not text:
            return {'success': False, 'error': 'Could not extract text'}
        
        sentences = FileProcessor.extract_sentences(text)
        entities = FileProcessor.extract_entities(text)
        
        # Remove duplicates
        unique_entities = []
        seen = set()
        for entity in entities:
            key = f"{entity['entity1']}_{entity['entity2']}"
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return {
            'success': True,
            'file_type': ext,
            'total_sentences': len(sentences),
            'text': text,
            'sentences': sentences[:10],
            'entities': unique_entities,
            'total_entities': len(unique_entities)
        }