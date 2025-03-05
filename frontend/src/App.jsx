import { useState } from 'react'



function App() {


  return (
    <div class = "flex flex-col items-center justify-center h-screen">
      <h1 class="text-4xl font-bold">Pose Detection</h1>

      <form id="upload-form" enctype="multipart/form-data">
        <input type="file" id="imageInput" name="image" accept="image/*" />
        <button
          type="submit"
          class="button bg-[#4a5568] text-white px-[10px] py-[15px] cursor-pointer"
        >
          Detect Pose
        </button>
      </form>

      <button
        id="startCameraButton"
        class="button bg-[#4a5568] text-white px-[10px] py-[15px] cursor-pointer"
      >
        Start Video
      </button>



      <h3 id="result-heading" class="hidden">Result:</h3>
      <img
        id="outputImage"
        class="fixed hidden h-[400px] w-[400px] left-[35vw]"
      />
    </div>

    // ????????
     
    // <h1 class="text-4xl font-bold">Pose Detection</h1>

    // <form id="upload-form" enctype="multipart/form-data">
    //   <input type="file" id="imageInput" name="image" accept="image/*" />
    //   <button
    //     type="submit"
    //     class="button bg-[#4a5568] text-white px-[10px] py-[15px] cursor-pointer"
    //   >
    //     Detect Pose
    //   </button>
    // </form>

    // <button
    //   id="startCameraButton"
    //   class="button bg-[#4a5568] text-white px-[10px] py-[15px] cursor-pointer"
    // >
    //   Start Video
    // </button>

    // <!-- <video id="video" autoplay class="w-[500px] h-[500px] hidden"></video>
    // <canvas id="outputCanvas" class="w-[500px] h-[500px] hidden"></canvas> -->

    // <h3 id="result-heading" class="hidden">Result:</h3>
    // <img
    //   id="outputImage"
    //   class="fixed hidden h-[400px] w-[400px] left-[35vw]"
    // />

    // <script>
    //   const form = document.getElementById("upload-form");
    //   const resultHeading = document.getElementById("result-heading");
    //   const outputImage = document.getElementById("outputImage");
    //   const startCameraButton = document.getElementById("startCameraButton");
    //   // const video = document.getElementById("video");
    //   // const canvas = document.getElementById("outputCanvas");
    //   // const context = canvas.getContext("2d");

    //   const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000/pose';

    //   function toggleVisibility(element, visible) {
    //     element.classList.toggle("hidden", !visible);
    //   }

    //   form.onsubmit = async (e) => {
    //     e.preventDefault();
    //     const imageFile = document.getElementById("imageInput").files[0];

    //     if (!imageFile) {
    //       alert("Please upload an image first.");
    //       return;
    //     }

    //     const reader = new FileReader();
    //     reader.onload = async () => {
    //       const base64Image = reader.result.split(",")[1];

    //       try {
    //         const response = await fetch(API_URL, {
    //           method: "POST",
    //           headers: { "Content-Type": "application/json" },
    //           body: JSON.stringify({ image: base64Image }),
    //         });

    //         if (!response.ok) throw new Error("Network response was not ok");

    //         const result = await response.json();
    //         if (result.image) {
    //           outputImage.src = `data:image/jpeg;base64,${result.image}`;
    //           toggleVisibility(outputImage, true);
    //           toggleVisibility(resultHeading, true);
    //         }
    //       } catch (error) {
    //         console.error("Fetch error:", error);
    //         alert("Failed to connect to the server. Please try again later.");
    //       }
    //     };
    //     reader.readAsDataURL(imageFile);
    //   };

      

    //   startCameraButton.onclick = () => {
    //     const popupWindow = window.open(
    //       "realtime.html", 
    //       "CameraWindow", 
    //       "width=800,height=600,top=100,left=100,resizable=yes,scrollbars=no"
    //     );

    //     if (!popupWindow ||
    //       popupWindow.closed ||
    //       typeof popupWindow.closed === "undefined"
    //     ){
    //       alert("Popup blocked! Please allow popups for this website.");
    //     }

      
    //   };
    // </script>


     
  )
}

export default App
