
import re
import numpy as np

def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # converte ndarray in lista
    elif isinstance(obj, dict):
        return {k: ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(i) for i in obj]
    else:
        return obj
    return obj

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
ANSI_COLOR_CODE_PATTERN = re.compile(r'\x1B\[(\d+)m')
ANSI_SPLIT_PATTERN = re.compile(r'(\x1B\[\d+m)([^\x1B]*)')
ANSI_COLOR_MAP = {
    '34': 'text-blue-600',    # Blue
    '37': 'text-white-700',    # White/Default (usiamo un grigio scuro)
    '93': 'text-yellow-600',  # Yellow (Alta Intensità)
    '33': 'text-yellow-900',  # Yellow (Bassa Intensità)
    '31': 'text-red-600',     # Red
    '91': 'text-red-900',     # Red (Alta Intensità)
    '32': 'text-green-600',   # Green
    '92': 'text-green-900',   # Green (Alta Intensità)
    '36': 'text-cyan-600',    # Cyan
    '96': 'text-cyan-900',    # Cyan (Alta Intensità)
    '0': 'text-gray-700'      # Reset/Default
}

def clean_ansi(text):
    """Rimuove i codici di escape ANSI da una stringa."""
    return ANSI_ESCAPE.sub('', text)

def convert_ansi_to_html(text: str) -> str:
    """
    Converte una stringa contenente codici di colore ANSI in un frammento HTML
    usando tag <span> e classi CSS di Tailwind.
    """
    
    # Inizializza il colore corrente a default (grigio scuro)
    current_class = ANSI_COLOR_MAP.get('0', 'text-gray-700')
    output_html = ""
    
    # Prepara il testo per l'analisi aggiungendo un codice di reset all'inizio se manca
    if not text.startswith('\x1B[') or not re.search(r'\x1B\[\d+m', text):
        text = '\x1B[0m' + text

    # Usa finditer per iterare su tutte le occorrenze di pattern [CODICE][TESTO]
    matches = ANSI_SPLIT_PATTERN.finditer(text)
    
    for match in matches:
        ansi_code_raw = match.group(1) # Es. '\x1B[32m'
        segment_text = match.group(2)  # Il testo che segue il codice
        
        # Estrae il codice numerico, es. '32'
        color_code_match = re.search(r'\d+', ansi_code_raw)
        if color_code_match:
            color_code = color_code_match.group(0)
            
            # Mappa il codice ANSI alla classe CSS o usa il default
            new_class = ANSI_COLOR_MAP.get(color_code, 'text-gray-700')
            
            # Aggiorna la classe corrente (per il prossimo segmento)
            current_class = new_class

            # Applica la classe al segmento di testo
            # Nota: html.escape() non è necessario qui perché il testo è un log semplice
            # e non contiene markup generato dall'utente, ma sarebbe buona pratica in generale.
            if segment_text:
                output_html += f'<span class="{current_class}">{segment_text}</span>'
                
    # Rimuove eventuali resti di ANSI all'inizio o alla fine se l'analisi non è stata perfetta
    # Se il testo iniziava con codice ANSI non catturato dal loop, questa pulizia finale è una safety net.
    return clean_ansi(output_html)