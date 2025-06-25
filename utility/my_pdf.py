import os
import json as js
from fpdf import FPDF

def create_pdf_from_directory(data, directory):
    # Create instance of FPDF class
    pdf = FPDF()
    
    # Add a page
    pdf.add_page()
    text_size=8
    section_size=10

    # Set title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, f"Execution {directory.replace('training/','')}", ln=True, align='C')
    
    # Add Config section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 30, f"Configuration", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', text_size)
    pdf.multi_cell(0, 5, js.dumps(data.config, separators=(',', ':'), indent=2))

    # Add Train Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 30, f"Train {data.train_excution_time} s", ln=True, align='C')
    
    images = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    pdf.set_font("Arial", 'B', section_size)
    pdf.cell(200, 15, f"Train metric", ln=True, align='C')
    pdf.set_font("Arial", 'B', text_size)
    pdf.cell(200, 10, f"accuracy", ln=True, align='C')
    pdf.multi_cell(0, 5, js.dumps(data.train_metrics["accuracy"], separators=(',', ':')))
    pdf.ln(10)  # Add line break
    pdf.cell(200, 10, f"precision", ln=True, align='C')
    pdf.multi_cell(0, 5, js.dumps(data.train_metrics["precision"], separators=(',', ':')))
    pdf.ln(10)  # Add line break
    pdf.cell(200, 10, f"recall", ln=True, align='C')
    pdf.multi_cell(0, 5, js.dumps(data.train_metrics["recall"], separators=(',', ':')))
    pdf.ln(10)  # Add line break
    pdf.cell(200, 10, f"f1_score", ln=True, align='C')
    pdf.multi_cell(0, 5, js.dumps(data.train_metrics["f1_score"], separators=(',', ':')))
    pdf.ln(10)  # Add line break
    image_path = os.path.join(directory, images[0])    
    pdf.image(image_path, w=pdf.w - 20) 

    pdf.add_page()        
    pdf.set_font("Arial", 'B', section_size)
    pdf.cell(200, 30, f"Train indicators", ln=True, align='C')
    pdf.set_font("Arial", 'B', text_size)
    pdf.multi_cell(0, 5, js.dumps(data.train_indicators, separators=(',', ':')))
    pdf.ln(10)  # Add line break
    image_path = os.path.join(directory, images[2])    
    pdf.image(image_path, w=pdf.w - 20) 
    
    if len(data.train_types["explorations"]) > 0 and len(data.train_types["exploitations"]) > 0:
        pdf.add_page()    
        pdf.set_font("Arial", 'B', section_size)
        pdf.cell(200, 30, f"Train types", ln=True, align='C')
        pdf.set_font("Arial", 'B', text_size)
        pdf.cell(200, 10, f"explorations", ln=True, align='C')
        pdf.multi_cell(0, 5, js.dumps(data.train_types["explorations"], separators=(',', ':')))
        pdf.ln(10)  # Add line break
        pdf.cell(200, 10, f"exploitations", ln=True, align='C')
        pdf.multi_cell(0, 5, js.dumps(data.train_types["exploitations"], separators=(',', ':')))
        pdf.ln(10)  # Add line break
        pdf.cell(200, 10, f"steps", ln=True, align='C')
        pdf.multi_cell(0, 5, js.dumps(data.train_types["steps"], separators=(',', ':')))
        pdf.add_page() 
        image_path = os.path.join(directory, images[1])    
        pdf.image(image_path, w=pdf.w - 20) 

    # Add Test Section
    if (False):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 30, f"Test {data.test_excution_time} s", ln=True, align='C')
        
        pdf.set_font("Arial", 'B', text_size)
        pdf.multi_cell(0, 5, js.dumps(data.test_indicators, separators=(',', ':'), indent=2))
        pdf.ln(10)  # Add line break
        pdf.multi_cell(0, 5, js.dumps(data.test_metrics, separators=(',', ':'), indent=2))

    # Save the PDF in the same directory
    pdf_output_path = os.path.join(directory, 'output.pdf')
    pdf.output(pdf_output_path)

    #print(f"PDF created successfully at {pdf_output_path}")

# Example usage
# Replace 'your_directory_path' with the actual directory path
#create_pdf_from_directory('your_directory_path')
