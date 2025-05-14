// Define variables in global scope
let session; // ONNX Session
let imagePredictions = []; // Array to store predictions from each image
const GROQ_API_KEY = "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"; // TODO: Move to env variables or secure config
// Track processed images to prevent duplicates
const processedImages = new Set();
// Store selected files
let selectedFiles = [];
// Store class mappings
let classMapping = {};
// Store body area selections
let bodyAreaSelections = {};
// Store photo dates
let photoDates = {};

// Define body area options
const bodyAreaOptions = [
  "Face", "Scalp", "Neck", "Chest", "Back", "Abdomen", "Arms",
  "Hands", "Legs", "Feet", "Genitals", "Buttocks", "Other"
];

const MODEL_FILENAME_TO_LOAD = "model2.onnx"; // Centralized model filename

// Theme functions
function changeTheme(theme) {
  document.body.className = `theme-${theme}`;
  localStorage.setItem('skinsavers-theme', theme);
}

// Loading screen functions
function showLoadingScreen(message = "Initializing...") {
  const loadingOverlay = document.getElementById("loading-overlay");
  const loadingStatus = document.getElementById("loading-status");
  const progressBar = document.getElementById("loading-progress-bar");
  if (loadingOverlay && loadingStatus && progressBar) {
    progressBar.style.width = "0%";
    loadingStatus.textContent = message;
    loadingOverlay.classList.add("active");
  } else {
    console.error("Loading screen elements not found!");
  }
}

function updateLoadingProgress(percent, message = null) {
  const progressBar = document.getElementById("loading-progress-bar");
  const loadingStatus = document.getElementById("loading-status");
  if (progressBar && loadingStatus) {
    progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`; // Clamp between 0 and 100
    if (message) {
      loadingStatus.textContent = message;
    }
  }
}

function hideLoadingScreen() {
  const loadingOverlay = document.getElementById("loading-overlay");
  if (loadingOverlay) {
    loadingOverlay.classList.remove("active");
  }
}

// Function to download and validate the ONNX model
async function validateAndGetModelData(url) {
  console.log(`Attempting to fetch and validate model from: ${url}`);
  try {
    const response = await fetch(url);
    if (!response.ok) {
      const absoluteUrl = new URL(url, window.location.origin).href;
      console.warn(`Failed to fetch model from ${absoluteUrl}: ${response.status} ${response.statusText}`);
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText} (URL: ${absoluteUrl})`);
    }
    const modelData = await response.arrayBuffer();
    const isValid = modelData.byteLength > 100000; // Basic check: is it at least 100KB? Adjust as needed.
    
    console.log(`Validation for ${url}: isValid=${isValid}, size=${modelData.byteLength}`);
    return { isValid, data: modelData, size: modelData.byteLength, url };
  } catch (error) {
    console.error(`Error fetching or validating model at ${url}:`, error);
    return { isValid: false, error: error.message, url };
  }
}

// Initialize file upload and preview functionality
document.addEventListener("DOMContentLoaded", function () {
  const savedTheme = localStorage.getItem('skinsavers-theme');
  if (savedTheme) {
    document.body.className = `theme-${savedTheme}`;
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) themeSelect.value = savedTheme;
  }

  const inputElement = document.getElementById("input-images");
  const previewContainer = document.getElementById("preview-container");
  const uploadArea = document.getElementById("upload-area");
  const bodyAreaContainer = document.getElementById("body-area-container");
  const bodyAreaSection = document.getElementById("body-area-section");

  if (!inputElement || !previewContainer || !uploadArea || !bodyAreaContainer || !bodyAreaSection) {
    console.error("One or more essential DOM elements for file upload are missing!");
    return;
  }

  bodyAreaSection.style.display = "none";

  inputElement.addEventListener("change", (e) => handleFiles(e.target.files));
  uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("dragover"); });
  uploadArea.addEventListener("dragleave", (e) => { e.preventDefault(); uploadArea.classList.remove("dragover"); });
  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
  });

  function handleFiles(incomingFiles) {
    const imageFiles = Array.from(incomingFiles).filter((file) => file.type.startsWith("image/"));
    if (imageFiles.length === 0) {
      alert("Please select image files only (e.g., JPG, PNG, WEBP).");
      return;
    }
    previewContainer.innerHTML = ""; // Clear existing previews
    bodyAreaContainer.innerHTML = ""; // Clear existing body area selections

    selectedFiles = []; // Reset global array
    bodyAreaSelections = {};
    photoDates = {};

    imageFiles.forEach((file) => {
      selectedFiles.push(file); // Add to our global list
      const reader = new FileReader();
      reader.onload = function (e) {
        const uniqueFileId = `${file.name}-${file.lastModified}-${file.size}`; // Create a more unique ID

        const previewDiv = document.createElement("div");
        previewDiv.className = "image-preview";
        previewDiv.dataset.fileId = uniqueFileId; // Store unique ID for removal

        const img = document.createElement("img");
        img.src = e.target.result;
        img.alt = `Preview of ${file.name}`;

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "remove-btn";
        removeBtn.innerHTML = "×";
        removeBtn.setAttribute("aria-label", `Remove ${file.name}`);
        removeBtn.addEventListener("click", function (event) {
          event.stopPropagation();
          
          // Find and remove the file from the main selectedFiles array using the unique ID
          selectedFiles = selectedFiles.filter(f => `${f.name}-${f.lastModified}-${f.size}` !== uniqueFileId);
          
          previewDiv.remove();
          const bodyAreaDomId = `body-area-item-${uniqueFileId.replace(/[^a-zA-Z0-9_.-]/g, '_')}`;
          const bodyAreaItem = document.getElementById(bodyAreaDomId);
          if (bodyAreaItem) bodyAreaItem.remove();
          
          delete bodyAreaSelections[uniqueFileId];
          delete photoDates[uniqueFileId];
          
          if (selectedFiles.length === 0) bodyAreaSection.style.display = "none";
        });
        previewDiv.appendChild(img);
        previewDiv.appendChild(removeBtn);
        previewContainer.appendChild(previewDiv);
        createBodyAreaSelection(file, uniqueFileId, e.target.result);
      };
      reader.onerror = () => {
        console.error(`Error reading file: ${file.name}`);
        alert(`Could not read file: ${file.name}. It might be corrupted.`);
      };
      reader.readAsDataURL(file);
    });

    if (imageFiles.length > 0) bodyAreaSection.style.display = "block";
  }

  function createBodyAreaSelection(file, uniqueFileId, imgSrc) {
    const domCompatibleId = uniqueFileId.replace(/[^a-zA-Z0-9_.-]/g, '_');
    const bodyAreaItem = document.createElement("div");
    bodyAreaItem.className = "body-area-item";
    bodyAreaItem.id = `body-area-item-${domCompatibleId}`;
    const thumbnail = document.createElement("img");
    thumbnail.src = imgSrc;
    thumbnail.className = "body-area-thumbnail";
    thumbnail.alt = `Thumbnail for ${file.name}`;
    const detailsDiv = document.createElement("div");
    detailsDiv.className = "body-area-details";
    const filenameDiv = document.createElement("div");
    filenameDiv.className = "body-area-filename";
    filenameDiv.textContent = file.name;
    const selectId = `body-area-select-${domCompatibleId}`;
    const selectLabel = document.createElement("label");
    selectLabel.setAttribute("for", selectId);
    selectLabel.textContent = "Body Area:";
    selectLabel.className = "sr-only"; 

    const select = document.createElement("select");
    select.className = "body-area-select";
    select.id = selectId;
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "-- Select Body Area --";
    select.appendChild(defaultOption);
    defaultOption.selected = true;

    bodyAreaOptions.forEach(area => {
      const option = document.createElement("option");
      option.value = area;
      option.textContent = area;
      select.appendChild(option);
    });
    select.addEventListener("change", function() { bodyAreaSelections[uniqueFileId] = this.value; });

    const dateInputId = `body-area-date-${domCompatibleId}`;
    const dateLabel = document.createElement("label");
    dateLabel.className = "date-label";
    dateLabel.setAttribute("for", dateInputId);
    dateLabel.textContent = "Photo Date (optional):";

    const dateInput = document.createElement("input");
    dateInput.type = "date";
    dateInput.className = "body-area-date";
    dateInput.id = dateInputId;
    const today = new Date();
    const formattedDate = today.toISOString().split('T')[0];
    dateInput.value = formattedDate;
    dateInput.addEventListener("change", function() { photoDates[uniqueFileId] = this.value; });
    photoDates[uniqueFileId] = formattedDate; 

    detailsDiv.appendChild(filenameDiv);
    detailsDiv.appendChild(selectLabel);
    detailsDiv.appendChild(select);
    detailsDiv.appendChild(dateLabel);
    detailsDiv.appendChild(dateInput);
    bodyAreaItem.appendChild(thumbnail);
    bodyAreaItem.appendChild(detailsDiv);
    bodyAreaContainer.appendChild(bodyAreaItem);
  }
});

// Load the ONNX model
async function loadModel() {
  try {
    showLoadingScreen("Loading model resources...");
    updateLoadingProgress(10, "Loading class mapping...");
    const mappingResponse = await fetch("class_mapping.json"); // Ensure this file exists in the same dir as index.html
    if (!mappingResponse.ok) throw new Error(`Failed to load class mapping: ${mappingResponse.status} ${mappingResponse.statusText}`);
    classMapping = await mappingResponse.json();
    console.log("Class mapping loaded:", classMapping);

    updateLoadingProgress(20, "Initializing ONNX runtime...");
    const ort = window.ort;
    ort.env = ort.env || {};
    ort.env.wasm = ort.env.wasm || {};
    ort.env.wasm.wasmPaths = { // Using a generally stable version
      'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort-wasm.wasm',
      'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort-wasm-simd.wasm',
      'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort-wasm-threaded.wasm',
      'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort-wasm-simd-threaded.wasm'
    };
    console.log("ONNX Runtime WebAssembly paths configured. ORT Version:", ort.version || "unknown (likely >=1.15.0)");

    updateLoadingProgress(40, "Validating model file...");
    // Assuming MODEL_FILENAME_TO_LOAD is relative to the HTML file's location
    const modelValidationResult = await validateAndGetModelData(MODEL_FILENAME_TO_LOAD);

    if (!modelValidationResult.isValid || !modelValidationResult.data) {
      console.error("Model validation failed or data is null:", modelValidationResult);
      throw new Error(modelValidationResult.error || `Failed to validate or find model: ${MODEL_FILENAME_TO_LOAD}`);
    }
    console.log(`Model ${MODEL_FILENAME_TO_LOAD} validated. Size: ${modelValidationResult.size} bytes.`);
    updateLoadingProgress(60, `Loading ${MODEL_FILENAME_TO_LOAD}...`);

    const sessionOptions = {
      executionProviders: ['wasm'], // 'wasm' is generally more robust than 'webgl' for complex models
      graphOptimizationLevel: 'all' // 'basic' or 'extended' might be safer if 'all' causes issues
    };
    
    session = await ort.InferenceSession.create(modelValidationResult.data, sessionOptions);
    console.log("ONNX session created successfully from ArrayBuffer for:", MODEL_FILENAME_TO_LOAD);

    if (session) {
      console.log("Model input names:", session.inputNames); // Should typically be ['input']
      console.log("Model output names:", session.outputNames); // Should typically be ['output']
    }

    updateLoadingProgress(90, "Finalizing setup...");
    setTimeout(() => {
      updateLoadingProgress(100, "Ready!");
      setTimeout(hideLoadingScreen, 500);
    }, 500);
  } catch (error) {
    console.error(`Critical Error in loadModel (${MODEL_FILENAME_TO_LOAD}):`, error);
    let userErrorMessage = `Failed to load the AI model (${MODEL_FILENAME_TO_LOAD}). Details: ${error.message}`;
    if (error.message.includes("protobuf") || error.message.includes("parse")) {
        userErrorMessage = `The model file (${MODEL_FILENAME_TO_LOAD}) could not be parsed. It might be corrupted, not a valid ONNX format, or an issue with the ONNX Runtime version.`;
    } else if (error.message.includes("404") || error.message.includes("Not Found") || error.message.includes("fetch model")) {
        userErrorMessage = `The model file (${MODEL_FILENAME_TO_LOAD}) was not found at its expected location. Please ensure it's in the correct web directory (usually same as index.html) and the name is exact.`;
    }
    updateLoadingProgress(100, "Model Load Error!");
    setTimeout(() => {
      hideLoadingScreen();
      const outputDiv = document.getElementById('output') || document.body;
      // Clear previous errors before showing new one
      const existingError = outputDiv.querySelector('.error-container');
      if (existingError) existingError.remove();
      
      outputDiv.insertAdjacentHTML('afterbegin', `<div class="error-container"><h3>Model Loading Failed</h3><p>${userErrorMessage}</p><p>Please try refreshing the page. If the problem persists, check the browser console (F12) for more technical details or contact support.</p></div>`);
    }, 1000);
  }
}

// Function to determine if a condition is cancerous
function isCancer(conditionName) {
  if (typeof conditionName !== 'string') return false;
  const cancerIndicators = ['melanoma', 'carcinoma', 'cancer', 'malignant'];
  const conditionLower = conditionName.toLowerCase();
  return cancerIndicators.some(indicator => conditionLower.includes(indicator));
}

// Preprocess image for ONNX model
async function preprocessImage(imageElement) {
  // Ensure TensorFlow.js is loaded
  if (typeof tf === 'undefined') {
    console.error("TensorFlow.js (tf) is not loaded!");
    throw new Error("Image processing library (TensorFlow.js) is not available.");
  }
  let imageTensor;
  try {
    imageTensor = tf.browser.fromPixels(imageElement);
    const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
    const normalizedImage = resizedImage.div(tf.scalar(255.0));
    const meanImageNet = tf.tensor([0.485, 0.456, 0.406]);
    const stdImageNet = tf.tensor([0.229, 0.224, 0.225]);
    const normalizedImageNet = normalizedImage.sub(meanImageNet).div(stdImageNet);
    const transposedImage = normalizedImageNet.transpose([2, 0, 1]).expandDims(0); // NCHW
    const imageData = await transposedImage.data(); // Float32Array
    
    // Explicitly dispose all created tensors
    imageTensor.dispose();
    resizedImage.dispose();
    normalizedImage.dispose();
    normalizedImageNet.dispose();
    transposedImage.dispose();
    
    return new Float32Array(imageData); // Ensure it's Float32Array
  } catch(error) {
    console.error("Error during image preprocessing with TensorFlow.js:", error);
    if (imageTensor) imageTensor.dispose(); // Attempt to dispose if created
    throw error; // Re-throw to be caught by processImage
  }
}

// Main function to process skin images
window.skinsave = async function () {
  if (!session) {
    alert("Model is still loading. Please wait or refresh the page if this message persists.");
    return;
  }
  if (selectedFiles.length === 0) {
    alert("Please select at least one image to analyze.");
    return;
  }
  const unselectedBodyAreas = selectedFiles.filter(file => {
    const uniqueFileId = `${file.name}-${file.lastModified}-${file.size}`;
    return !bodyAreaSelections[uniqueFileId] || bodyAreaSelections[uniqueFileId] === "";
  });
  if (unselectedBodyAreas.length > 0) {
    alert(`Please select body areas for all images. Missing for: ${unselectedBodyAreas.map(f=>f.name).join(', ')}`);
    return;
  }

  showLoadingScreen("Preparing for analysis...");
  updateLoadingProgress(5); // Start with a small progress
  
  console.log("Processing images...", selectedFiles.length, "files total.");
  document.getElementById("output").innerHTML = ""; // Clear previous results
  imagePredictions = []; // Reset predictions for this run
  processedImages.clear(); // Clear processed set for this run

  const totalImages = selectedFiles.length;
  for (let i = 0; i < totalImages; i++) {
    const currentFile = selectedFiles[i];
    // Progress from 5% to 65% for image processing part
    const progressPercent = 5 + Math.round(((i + 1) / totalImages) * 60); 
    updateLoadingProgress(progressPercent, `Analyzing image ${i+1} of ${totalImages}: ${currentFile.name}`);
    console.log(`Starting processing for image ${i+1}: ${currentFile.name}`);
    await processImage(currentFile);
  }

  // Store predictions for Groq
  document.getElementById("groq-data").textContent = JSON.stringify(imagePredictions);
    
  updateLoadingProgress(65, "Generating comprehensive analysis via AI Assistant...");
  await generateCancerAdvice(); // This will update progress further
  
  hideLoadingScreen(); // Hide loading screen after Groq advice
};

// Function to process each image and display predictions
async function processImage(inputFile) {
  const uniqueFileId = `${inputFile.name}-${inputFile.lastModified}-${inputFile.size}`;
  if (processedImages.has(uniqueFileId)) {
    console.log(`Skipping already processed image: ${inputFile.name}`);
    return;
  }
  processedImages.add(uniqueFileId);

  const imageElement = document.createElement("img");
  const resultContainer = document.createElement("div"); // Container for this image's output
  resultContainer.classList.add("result-container");
  // Prepend image to result container so it appears above the text results
  resultContainer.appendChild(imageElement); 
  document.getElementById("output").appendChild(resultContainer); // Add to DOM early for visual feedback

  // Create a div for text results (predictions or errors)
  const resultDiv = document.createElement("div"); 
  resultDiv.classList.add("prediction-details");
  resultContainer.appendChild(resultDiv); // Add text results div below image

  try {
    imageElement.src = URL.createObjectURL(inputFile);
    await new Promise((resolve, reject) => {
        imageElement.onload = resolve;
        imageElement.onerror = () => reject(new Error(`Failed to load image data for ${inputFile.name}. The file might be corrupted.`));
    });

    console.log("Image element loaded for ONNX:", inputFile.name);
    const imageData = await preprocessImage(imageElement); // Preprocess
    
    // Ensure input name matches what the model expects (usually 'input')
    const inputName = session.inputNames[0]; 
    const inputTensor = new window.ort.Tensor('float32', imageData, [1, 3, 224, 224]);
    const feeds = { [inputName]: inputTensor }; 
    
    console.log(`Running inference for ${inputFile.name} with input: ${inputName}`);
    const results = await session.run(feeds);
    
    const outputName = session.outputNames[0]; // Ensure output name matches
    const outputTensor = results[outputName];
    if (!outputTensor || !outputTensor.data) {
        throw new Error(`Model output '${outputName}' is missing or invalid.`);
    }
    const outputData = outputTensor.data;
    const softmaxData = softmax(Array.from(outputData)); // Convert to probabilities
    console.log(`Softmax probabilities for ${inputFile.name}:`, softmaxData);

    const formattedPredictions = [];
    for (let i = 0; i < softmaxData.length; i++) {
      const className = classMapping[i] || `Unknown Class ${i}`; // Graceful fallback
      const isCancerous = isCancer(className);
      formattedPredictions.push({ className, probability: softmaxData[i], isCancer: isCancerous });
    }
    formattedPredictions.sort((a, b) => b.probability - a.probability); // Sort by probability
    console.log(`Formatted predictions for ${inputFile.name}:`, formattedPredictions);

    const bodyArea = bodyAreaSelections[uniqueFileId] || "Not specified";
    const photoDate = photoDates[uniqueFileId] || "Not specified";
    resultDiv.innerHTML = `<b>Prediction for ${inputFile.name} (Area: ${bodyArea}, Date: ${photoDate}):</b><br>`;
    
    formattedPredictions.slice(0, 5).forEach((pred) => { // Show top 5 predictions
      const probabilityPercentage = (pred.probability * 100).toFixed(1);
      const colorClass = pred.isCancer ? "melanoma" : "benign"; // CSS classes for styling
      resultDiv.innerHTML += `
         <div class="progress-container">
           <div class="progress-label">
             <span>${pred.className} ${pred.isCancer ? '<strong class="cancer-flag">(POTENTIAL CANCER)</strong>' : '(likely benign)'}</span>
             <span>${probabilityPercentage}%</span>
           </div>
           <div class="progress-bar">
             <div class="progress-fill ${colorClass}" style="width: ${Math.min(100, parseFloat(probabilityPercentage))}%"></div>
           </div>
         </div>`;
    });
    const cancerRisk = formattedPredictions.filter(p => p.isCancer).reduce((s, p) => s + p.probability, 0) * 100;
    resultDiv.innerHTML += `<div class="cancer-risk"><strong>Model's Estimated Cancerous Indication Risk: ${cancerRisk.toFixed(1)}%</strong></div>`;
    
    imagePredictions.push({
      imageName: inputFile.name,
      uniqueFileId: uniqueFileId, // Store for reference
      bodyArea: bodyArea,
      photoDate: photoDate,
      predictions: formattedPredictions.slice(0,3), // Send top 3 to Groq for brevity
      cancerRisk: cancerRisk
    });

  } catch (error) {
    console.error(`Error processing image ${inputFile.name}:`, error);
    resultDiv.innerHTML = `<div class="error-message"><h4>Error processing ${inputFile.name}</h4><p>${error.message}</p><p>Please ensure the image is clear and the model is loaded correctly. Try a different image or refresh the page.</p></div>`;
  } finally {
      if (imageElement.src.startsWith('blob:')) { // Only revoke if it's an object URL
        URL.revokeObjectURL(imageElement.src); // Clean up object URL
      }
  }
}

// Softmax function
function softmax(arr) {
  if (!arr || arr.length === 0) return [];
  const maxVal = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - maxVal));
  const sumExps = exps.reduce((acc, curr) => acc + curr, 0);
  if (sumExps === 0) return arr.map(() => 1 / arr.length); // Avoid division by zero, distribute probability
  return exps.map(exp => exp / sumExps);
}

// Direct API call to Groq
async function callGroqAPI(messages) {
  if (!GROQ_API_KEY || GROQ_API_KEY === "YOUR_GROQ_API_KEY_HERE") {
    console.error("Groq API Key is not set. Please configure it securely.");
    throw new Error("Groq API Key is missing or not configured.");
  }

  const currentModel = "deepseek-r1-distill-llama-70b"

  console.log(`Attempting to call Groq API with model: ${currentModel}`);

  try {
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: currentModel, // Use the chosen model
        messages: messages,
        temperature: 0.6,    // Controls randomness: lower is more deterministic
        max_tokens: 4096,    // Max tokens to generate in the response
        top_p: 0.95,         // Nucleus sampling: considers tokens with top_p probability mass
        // stream: false,    // Set to true if you want to handle streaming responses
      }),
    });

    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json(); // Try to parse JSON error from Groq
      } catch (e) {
        // If parsing fails, use the raw response text
        const errorText = await response.text();
        console.error("Groq API raw error response:", errorText);
        errorData = { error: { message: `Non-JSON error response: ${response.statusText || "Unknown API error"}. Raw: ${errorText.substring(0, 200)}...` } };
      }
      console.error("Groq API error data:", errorData);
      throw new Error(
        `Groq API error: ${response.status} - ${errorData.error?.message || response.statusText || "Unknown error from API"}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error during Groq API call:", error);
    // Re-throw the error so the calling function (generateCancerAdvice) can handle it
    // and display a user-friendly message.
    throw error;
  }
}
// Generate cancer advice using Groq AI
async function generateCancerAdvice() {
  const predictionDataForGroq = JSON.parse(document.getElementById("groq-data").textContent);
  if (!predictionDataForGroq || predictionDataForGroq.length === 0) {
    console.log("No prediction data to send to Groq.");
    updateLoadingProgress(100, "No image data to analyze.");
    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML += "<p>No image predictions were generated. Please upload images and try again.</p>";
    return;
  }
  console.log("Prediction data for Groq:", predictionDataForGroq);

  const prompt = `
     Based on the following skin lesion image analysis results, provide a comprehensive analysis. Each item in "Predictions Data" corresponds to one uploaded image.

     Predictions Data: 
     ${JSON.stringify(predictionDataForGroq, null, 2)}

     Please structure your response into the following sections:

     1. OVERVIEW OF FINDINGS:
        - For each image analyzed (refer to by imageName), summarize the top predicted lesion type and its confidence (probability).
        - Highlight any predictions that are particularly concerning (e.g., high confidence for melanoma or basal cell carcinoma).

     2. DETAILED ADVICE PER CONCERNING LESION:
        - For each lesion identified as potentially cancerous or highly suspicious by the model:
            - Detected Condition: [e.g., Melanoma (model confidence: X%)] on [Body Area] (Photo Date: [Date, if provided])
            - General Information: Briefly describe this condition.
            - Recommended Actions: What are the immediate next steps? (e.g., Consult a dermatologist urgently).
            - Monitoring: What signs of change should be monitored for this type of lesion on this body area?
            - Lifestyle Considerations: Any relevant lifestyle advice.

     3. GENERAL SKIN HEALTH AND MONITORING RECOMMENDATIONS:
        - Based on the overall findings, provide general advice for skin health.
        - Suggest a follow-up or monitoring schedule if multiple concerning areas were found or if there's a high overall "Model's Estimated Cancerous Indication Risk."
        - Briefly discuss the importance of regular self-examinations and professional dermatological check-ups, especially considering the findings.

     4. INTERPRETING MODEL RESULTS (EDUCATIONAL):
        - Explain that the model provides predictions based on patterns and its confidence indicates similarity to training data for that class.
        - Clarify that this is not a diagnosis. High confidence for a benign condition (e.g., Nevus) is reassuring but not definitive. High confidence for a malignant condition is a strong signal to seek professional medical evaluation.
        - If photo dates were provided and show changes, explain how a doctor might use this information.

     Rough Prediction of Stage/Progression of Cancer (Model Interpretation): 
     This section should interpret the model's confidence percentages as indicators of how closely a lesion matches the characteristics of a given condition. For example, a high percentage for melanoma suggests strong resemblance. If dates are available for lesions showing concerning signs, discuss how a physician might interpret this in terms of changes. Do not attempt to assign a specific medical stage (e.g., Stage I-IV). Briefly explain potential timelines for seeking medical advice based on the findings. Add the exact disclaimer: "This prediction should not be used for potential life altering decisions, and should only be used for casual advice."
     
     IMPORTANT INSTRUCTIONS:
     1. Start your response with the heading "Skin Lesion Analysis and Recommendations".
     2. Refer to yourself as "assistant".
     3. Format in a professional, clinical manner with clear sections and bullet points.
     4. Do not include any disclaimers about not being a medical professional, EXCEPT for the specific one in the "Rough Prediction" section.
     5. If the model predicts 'melanoma' or 'basal cell carcinoma' with significant confidence (e.g., >20-30%), strongly emphasize the need for immediate medical consultation for those specific lesions.
     6. Be direct and provide actionable information.
     7. Use bold text for subheadings within sections (e.g., **Detected Condition:**). Do not use #.
     8. If photo dates are available for multiple images, comment on any patterns a physician might look for if these represented the same lesion over time (but state the model assumes they are different lesions unless told otherwise).
   `;

  const messages = [
    { role: "system", content: "You are a specialized medical assistant providing information based on skin lesion image analysis data. Your goal is to offer comprehensive insights on potential conditions, treatment considerations, and monitoring advice in a professional and clinical manner. Refer to yourself as 'assistant'." },
    { role: "user", content: prompt },
  ];

  try {
    updateLoadingProgress(75, "AI Assistant is analyzing results..."); // Progress update
    console.log("Sending prompt to Groq AI Assistant...");
    
    const apiResponse = await callGroqAPI(messages);
    let aiResponse = apiResponse.choices[0].message.content;
    
    updateLoadingProgress(90, "Formatting AI Assistant's response...");
    console.log("Analysis response received from Groq.");
    
    aiResponse = trimResponseToHeading(aiResponse, "Skin Lesion Analysis and Recommendations");
    const formattedResponse = formatProfessionalResponse(aiResponse); // Use your formatting function
    
    const resultContainer = document.createElement("div");
    resultContainer.classList.add("chat-output");
    resultContainer.innerHTML = formattedResponse;
    document.getElementById("output").appendChild(resultContainer); // Append Groq response

    // Add the main disclaimer at the very end of all outputs
    const finalDisclaimer = document.createElement("div");
    finalDisclaimer.className = "disclaimer final-disclaimer"; // Add specific class for styling
    finalDisclaimer.innerHTML = "<strong>Overall Disclaimer:</strong> This entire analysis, including AI assistant feedback, is for informational purposes only and does not constitute medical advice. Always consult with a qualified healthcare provider for any health concerns, diagnosis, and treatment.";
    document.getElementById("output").appendChild(finalDisclaimer);
    
    updateLoadingProgress(100, "Analysis complete!");
  } catch (error) {
    console.error("Error with Groq AI Assistant analysis:", error);
    updateLoadingProgress(100, "Error in AI Assistant Analysis!");
    const errorContainer = document.createElement("div");
    errorContainer.classList.add("chat-output", "error-container"); // Add error-container class
    errorContainer.innerHTML = `<h3>AI Assistant Analysis Error</h3><p>Sorry, there was an error generating the detailed analysis. Please try again later.</p><p>Details: ${error.message}</p>`;
    document.getElementById("output").appendChild(errorContainer);
  }
}

// Trim the response to start with a specific heading
function trimResponseToHeading(text, headingText) {
  if (typeof text !== 'string') return '';
  const headingPattern = new RegExp(`^\\s*${headingText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'im'); // Case-insensitive, multiline, start of line
  const match = text.match(headingPattern);
  if (match) return text.substring(match.index); // Return from the heading onwards
  console.warn(`Heading "${headingText}" not found in AI response. Returning full response.`);
  return text; // Return original if heading not found
}

// Format the response in a professional, clinical manner
function formatProfessionalResponse(text) {
  if (typeof text !== 'string') return '';
  let html = text;

  // Main Heading
  html = html.replace(/(Skin Lesion Analysis and Recommendations)/i, '<h1 class="analysis-header">$1</h1>');

  // Section Headings (e.g., "1. OVERVIEW OF FINDINGS:")
  // This regex looks for a number, dot, space, then uppercase words possibly with spaces, ending with a colon.
  html = html.replace(/^(\d+\.\s+[A-Z\s]+:)/gm, (match) => {
      const headerText = match.replace(/^\d+\.\s*/, '').replace(/:$/, '').trim();
      return `</div><div class="analysis-section"><h2>${headerText}</h2>`;
  });
  // Clean up potential leading </div>
  if (html.startsWith('</div>')) {
      html = html.substring(6);
  }
  // Ensure the first section also gets wrapped if not already
  const firstH2Index = html.indexOf('<h2>');
  const firstH1Index = html.indexOf('<h1 class="analysis-header">');
  if (firstH2Index > -1 && (firstH1Index === -1 || firstH2Index > html.indexOf('</h1>', firstH1Index))) {
      const partBeforeH2 = html.substring(0, firstH2Index);
      const partFromH2 = html.substring(firstH2Index);
      if (!partBeforeH2.trim().endsWith('</div class="analysis-section">') && !partBeforeH2.includes('<div class="analysis-section"')) {
         html = partBeforeH2 + '<div class="analysis-section">' + partFromH2;
      }
  }
  html += '</div>'; // Close the last section

  // Subheadings (Bold text, often on its own line or followed by a colon)
  // This regex tries to match markdown-style bold or lines that look like subheadings
  html = html.replace(/^\s*\*\*(.*?)\*\*\s*$/gm, '<div class="subheading">$1</div>'); // **Bold on its own line**
  html = html.replace(/^(?!<h[12]>)([A-Za-z\s]+:)\s*$/gm, (match, p1) => { // Text ending with colon, not already h1/h2
    if (p1.toUpperCase() === p1 && p1.length > 5) { // Heuristic: if all caps and somewhat long
        return `<div class="subheading">${p1}</div>`;
    }
    return match; // leave as is if not matching heuristic
  });


  // Bullet points (lines starting with - or *)
  // Convert multiple bullet points into a single <ul>
  html = html.replace(/^(?:\s*-\s+|\s*\*\s+)(.*?)(?:\n|$)/gm, '<li>$1</li>');
  // Wrap consecutive <li> tags with <ul>
  let listWrappedHtml = "";
  let inList = false;
  html.split('\n').forEach(line => {
      if (line.trim().startsWith('<li>')) {
          if (!inList) {
              listWrappedHtml += '<ul>\n';
              inList = true;
          }
          listWrappedHtml += line + '\n';
      } else {
          if (inList) {
              listWrappedHtml += '</ul>\n';
              inList = false;
          }
          listWrappedHtml += line + '\n';
      }
  });
  if (inList) { // Close any trailing list
      listWrappedHtml += '</ul>\n';
  }
  html = listWrappedHtml.trim();


  // Paragraphs: Wrap lines that aren't headings, list items, or already in divs, into <p> tags
  // This is complex to do perfectly with regex. A simpler approach might be to split by \n\n
  // For now, let's assume Groq provides reasonable paragraphing or we handle it with CSS whitespace.
  // A more robust solution would parse the content more structurally.

  // Specific Disclaimer for "Rough Prediction" section
  html = html.replace(
    /(This prediction should not be used for potential life altering decisions, and should only be used for casual advice\.)/g,
    '<p class="conclusion-disclaimer">$1</p>' // Wrap in paragraph with class
  );

  return html;
}

// Initialize the model when the page loads
window.addEventListener("load", loadModel);
