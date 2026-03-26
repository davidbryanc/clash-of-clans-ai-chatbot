# parse_xml.py (VERSI 2.1 - dengan penanganan sel kosong)

import os
import xml.etree.ElementTree as ET
import wikitextparser as wtp
import re

# =====================================================================
# KONFIGURASI
# =====================================================================
XML_FILE_PATH = "coc_export1.xml"
OUTPUT_DIR = "parsed_data"
# =====================================================================

def parse_wikitable_to_text(table):
    """
    Mengubah objek tabel wikitextparser menjadi teks yang bisa dibaca manusia.
    """
    text_lines = []
    headers = []
    
    table_data = table.data()
    if not table_data:
        return "" # Kembalikan string kosong jika tabel tidak punya data

    # Coba ambil header dari baris pertama
    first_row = table_data[0]
    headers = [h.strip() if h is not None else "" for h in first_row]

    # Loop melalui sisa baris data
    for i, row in enumerate(table_data):
        if i == 0: continue

        # --- PERBAIKAN UTAMA ADA DI SINI ---
        # Memeriksa apakah sel itu None sebelum melakukan .strip()
        row_data = [cell.strip() if cell is not None else "" for cell in row]
        # -----------------------------------
        
        line_parts = []
        for j, cell_value in enumerate(row_data):
            if j < len(headers) and headers[j] and cell_value:
                clean_header = re.sub(r'\{\{.*?\}\}', '', headers[j]).strip()
                if clean_header: # Pastikan header tidak kosong setelah dibersihkan
                    line_parts.append(f"{clean_header}: {cell_value}")
        
        if line_parts:
            text_lines.append(", ".join(line_parts))
            
    return "\n".join(text_lines)


def clean_wikitext(text):
    """
    Membersihkan wikitext, sekarang dengan penanganan tabel statistik.
    """
    if text is None:
        return "" # Pengaman tambahan jika teks utama kosong

    parsed = wtp.parse(text)
    
    stats_text_parts = []
    for table in parsed.tables:
        if 'Level' in str(table) or 'Hitpoints' in str(table):
            stats_text_parts.append("\n[Statistic Table]\n" + parse_wikitable_to_text(table))
    
    for table in parsed.tables:
        text = text.replace(str(table), "")
        
    parsed = wtp.parse(text)
    general_text = parsed.plain_text()
    
    full_text = general_text + "\n\n" + "\n".join(stats_text_parts)
    
    full_text = re.sub(r'==.*?==', '', full_text)
    full_text = re.sub(r'^\s*[^\w\s]+\s*$', '', full_text, flags=re.MULTILINE)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    return full_text.strip()


def main():
    if not os.path.exists(XML_FILE_PATH):
        print(f"ERROR: File '{XML_FILE_PATH}' not found.")
        return
        
    print(f"Reading and parsing XML file: '{XML_FILE_PATH}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    context = ET.iterparse(XML_FILE_PATH, events=('end',))
    page_count = 0
    
    try:
        _, root = next(context)
        namespace = root.tag.split('}')[0].strip('{')
    except StopIteration:
        print("ERROR: XML file seems to be empty or malformed.")
        return

    for event, elem in context:
        if elem.tag == f"{{{namespace}}}page":
            title_elem = elem.find(f"{{{namespace}}}title")
            text_elem = elem.find(f".//{{{namespace}}}text")
            
            if title_elem is not None and text_elem is not None:
                title = title_elem.text
                wikitext = text_elem.text
                
                if title and ":" not in title:
                    print(f"  -> Processing page: {title}")
                    plain_text = clean_wikitext(wikitext)
                    
                    if plain_text:
                        safe_filename = title.replace('/', '_').replace('\\', '_')
                        filename = f"{safe_filename}.txt"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(plain_text)
                        page_count += 1
            
            elem.clear()

    print(f"\n✅ Success! Processed {page_count} pages and saved them to the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    main()
