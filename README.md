# Blockchain-Governed-Federated-Learning-with-Secure-Computation-for-Privacy-Preserving-EHR-Analytics
To Proposed a Blockchain-Governed Federated Learning with Secure Third-Party Computation (BFL-STPC) framework for secure and efficient collaborative healthcare analytics

Overview : 
BFL-STPC is a proposed privacy-preserving collaborative analytics framework for Electronic Health Records (EHRs) in distributed healthcare environments. It combines Federated Learning (FL), blockchain-based governance, and a Secure Third-Party Computation (STPC) module to support multi-hospital model training without centrally pooling raw patient records. The proposed STPC layer incorporates homomorphic encryption (HE), secure multi-party computation (SMPC), secure aggregation, and differential privacy (DP). Blockchain smart contracts are intended to support participant authentication, access authorization, model-update traceability, and tamper-resistant audit records.

Proposed System Architecture: 
Hospital A (local EHRs) ──> Local training ──> Protected model update ──┐
Hospital B (local EHRs) ──> Local training ──> Protected model update ──┼──> STPC / secure aggregation
Hospital C (local EHRs) ──> Local training ──> Protected model update ──┘                │
                                                                                     ▼
                                                                           Global model update
                                                                                     │
                                                                                     ▼
                                                                       Distributed to hospitals

Blockchain governance: participant authorization · policy enforcement ·
                       model-update metadata/provenance · audit events

STPC mechanisms (as proposed): HE · SMPC · secure aggregation · DP

Dataset : 
A synthetic healthcare dataset with 55,500 patient records was used to assess the suggested BFL-STPC framework [26]. The data set contains many attributes, including age, gender, blood type, medical condition, admission type, drug prescribed, laboratory results, insurance provider and amount billed. The cleaned records were then subjected to machine-learning classification task after the preprocessing and discarding of the duplicates. The dataset was split into training and testing sets with an 80:20 ratio technique to ensure that every class was represented on both sets. In the federated learning experiment, the patient information was split into 10 simulated hospitals, with each hospital as an FL client and having their patient data stored locally. The manuscript clearly states the data allocation follows independent and identically distributed (IID) or non-independent and identically distributed (non-IID) distribution. Local training is done independently in each hospital without sharing raw data. The proposed BFL-STPC framework provides secure communication of model updates for global aggregation, which only authorized updates are communicated, thus ensuring the privacy of locally stored patient information.

Experimental Environment: 
The proposed BFL-STPC framework was implemented in an Amazon Web Services (AWS) EC2 cloud environment, primarily using AWS P3-family instances equipped with NVIDIA Tesla V100 GPUs for model training, while AWS G4dn instances with NVIDIA T4 GPUs supported smaller-scale experiments. The framework was developed using Python and PyTorch to simulate distributed hospital nodes and a centralized federated aggregation server. Privacy-preserving computation was supported through CKKS-based Homomorphic Encryption (HE), Secure Multi-Party Computation (SMPC) using threshold secret sharing, and Differential Privacy (DP). Each hospital trained its model locally and transmitted only privacy-protected model updates for secure global aggregation.




























