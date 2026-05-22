from scipy.io import loadmat


####################
# Start of main function
####################

# Load the .mat file
file_path = 'C:/Users/phde5146/OneDrive - The University of Sydney (Staff)/aa_PdeC_Usyd/Teaching/BiomedCourses/BMET3997_Biological Digital Signal Analysis/2025/MajorProject/ProjectTrainData.mat'

mat_contents = loadmat(file_path)

ECG = mat_contents['ECG']
QRSexpert = mat_contents['QRSexpert']

#fix up the structures of ECG and QRSexpert by making them list of NumPy arrays

ECG=ECG.flatten().tolist()
for i in range(len(ECG)):
     ECG[i]=ECG[i].flatten()

QRSexpert=QRSexpert.flatten().tolist()
for i in range(len(QRSexpert)):
     QRSexpert[i]=QRSexpert[i].flatten()
     