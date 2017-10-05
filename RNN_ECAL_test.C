void RNN_ECAL_test() {

   
   TFile *input(0);

   TString fname = "EcalData.root";
   input = TFile::Open( fname ); // check if file in local directory exists
 
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }

   std::cout << "--- RNNClassification  : Using input file: " << input->GetName() << std::endl;


  // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "TMVA_DNN.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Creating the factory object
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );
   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");

   TTree *signalTree     = (TTree*)input->Get("sig");
   TTree *background     = (TTree*)input->Get("bgk");


   signalTree->Print();

   background->Print();

// add variables (time zero and time 1)
for (int i = 300; i < 700; ++i) {
    TString varName = TString::Format("EB_adc0[%d]",i);
    dataloader->AddVariable(varName,'F');
}

for (int i = 300; i < 700; ++i) {
    TString varName = TString::Format("EB_adc1[%d]",i);
    dataloader->AddVariable(varName,'F');
}


dataloader->AddSignalTree    ( signalTree,     1.0 );
dataloader->AddBackgroundTree( background,   1.0 );

// check given input
auto & datainfo = dataloader->GetDataSetInfo();
auto vars = datainfo.GetListOfVariables(); 
std::cout << "number of variables is " << vars.size() << std::endl;
for ( auto & v : vars) std::cout << v << ","; 
std::cout << std::endl;

int ntrainEvts = 500;
int ntestEvts =  500; 
TString trainAndTestOpt = TString::Format("nTrain_Signal=%d:nTrain_Background=%d:nTest_Signal=%d:nTest_Background=%d:SplitMode=Random:NormMode=NumEvents:!V",ntrainEvts,ntrainEvts,ntestEvts,ntestEvts );
TCut mycuts = "";//Entry$<1000"; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
 TCut mycutb = "";//Entry$<1000"; 
dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,trainAndTestOpt);

std::cout << "prepared DATA LOADER " << std::endl;
   // Input Layout
   TString inputLayoutString("InputLayout=1|1|300");

   // Batch Layout
   TString batchLayoutString("BatchLayout=256|2|300");

   // General layout.
   TString layoutString ("Layout=RNN|128|300|2|0,DENSE|64|TANH,DENSE|2|LINEAR");

   // Training strategies.
   TString training0("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=100,BatchSize=256,TestRepetitions=1,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");
   TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString training2("LearningRate=1e-3,Momentum=0.0,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString trainingStrategyString ("TrainingStrategy=");
   trainingStrategyString += training0; // + "|" + training1 + "|" + training2;

   // General Options.
   TString rnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
                       "WeightInitialization=XAVIERUNIFORM");

   rnnOptions.Append(":"); rnnOptions.Append(inputLayoutString);
   rnnOptions.Append(":"); rnnOptions.Append(batchLayoutString);
   rnnOptions.Append(":"); rnnOptions.Append(layoutString);
   rnnOptions.Append(":"); rnnOptions.Append(trainingStrategyString);
   rnnOptions.Append(":Architecture=CPU");

   factory->BookMethod(dataloader, TMVA::Types::kDL, "DNN_CPU", rnnOptions);


 factory->TrainAllMethods();


}
