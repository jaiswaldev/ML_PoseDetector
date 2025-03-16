import { useState } from 'react';
import PropTypes from 'prop-types';

const FileUploadForm = ({ onSuccess, onError, loading, setLoading }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const API_URL = import.meta.env.VITE_API_URL;

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        onError('Please upload an image file');
        return;
      }
      setSelectedFile(file);
      onError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      onError('Please select a file first');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      onSuccess(result);
      onError(null);
    } catch (error) {
      console.error('Upload error:', error);
      onError(error.message || 'Failed to upload image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form 
      onSubmit={handleSubmit}
      className="w-full max-w-md flex flex-col items-center gap-4 "
    >
      <div className="w-full">
        <label 
          htmlFor="imageInput"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          Upload Image
        </label>
        <input
          type="file"
          id="imageInput"
          accept="image/*"
          onChange={handleFileChange}
          disabled={loading}
          className="block w-full text-sm text-gray-500 items-center
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-[#4a5568] file:text-white
            hover:file:bg-[#2d3748]"
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className={`bg-[#4a5568] text-white px-4 py-2 rounded 
          hover:bg-[#2d3748] transition-colors
          ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        {loading ? 'Processing...' : 'Detect Pose'}
      </button>
    </form>
  );
};

FileUploadForm.propTypes = {
  onSuccess: PropTypes.func.isRequired,
  onError: PropTypes.func.isRequired,
  loading: PropTypes.bool.isRequired,
  setLoading: PropTypes.func.isRequired,
};

export default FileUploadForm;