import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import datetime

def create_density_report():
    filename = "Adversarial_Distribution_Report.pdf"
    
    with PdfPages(filename) as pdf:
        # Page 1
        plt.figure(figsize=(8.5, 11))
        
        plt.subplot(2, 1, 1)
        plt.axis('off')
        plt.text(0.5, 0.5, "Adversarial Regression\nDistribution Analysis", 
                 ha='center', va='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.3, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}",
                 ha='center', va='center', fontsize=12)
        
        # Add Density Plot
        try:
            img = mpimg.imread('density_results.png')
            plt.subplot(2, 1, 2)
            plt.imshow(img)
            plt.axis('off')
            plt.title("Kernel Density Estimate (KDE)", fontsize=14)
        except:
            pass
            
        pdf.savefig()
        plt.close()
        
        # Page 2
        plt.figure(figsize=(8.5, 11))
        
        # Add Histogram Plot
        try:
            img = mpimg.imread('histogram_results.png')
            plt.subplot(2, 1, 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title("Histogram Comparison", fontsize=14)
        except:
            pass
            
        plt.subplot(2, 1, 2)
        plt.axis('off')
        concl = (
            "Analysis:\n"
            "The KDE plot compares the smooth probability density of the\n"
            "real target values versus the GAN-generated values.\n\n"
            "- Good Fit: The curves should overlap significantly.\n"
            "- Mode Collapse: If Green is much narrower than Blue.\n"
            "- Offset: If the peaks are misaligned.\n"
        )
        plt.text(0.1, 0.8, concl, ha='left', va='top', fontsize=12, family='monospace')
        
        pdf.savefig()
        plt.close()

    print(f"Report generated: {filename}")

if __name__ == "__main__":
    create_density_report()
