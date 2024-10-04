import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);

    const onFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            if (selectedFile.type.startsWith('audio/')) { // Ensure it's an audio file
                setFile(selectedFile);
                setResult(''); // Clear result on new file selection
            } else {
                alert("Please select an audio file!");
            }
        }
    };

    const onFileUpload = async () => {
        if (!file) {
            alert("Please select a file first!");
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            setLoading(true);
            const response = await axios.post('http://localhost:5000/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            setLoading(false);
            if (response.data && response.data.result) {
                setResult(response.data.result);
            } else {
                setResult("Unexpected response from server");
            }
        } catch (error) {
            setLoading(false);
            console.error("Error uploading the file:", error);

            if (error.response && error.response.data) {
                alert("Error: " + JSON.stringify(error.response.data));
            } else if (error.request) {
                alert("Error: No response received from the server. Please try again.");
            } else {
                alert("Error: " + error.message);
            }
        }
    };

    return (
        <div>
            <h1>Upload Audio File for Deepfake Detection</h1>
            <input type="file" onChange={onFileChange} accept="audio/*" />
            <button onClick={onFileUpload} disabled={loading}>
                {loading ? 'Analyzing...' : 'Upload and Analyze'}
            </button>
            {result && <h3>Result: {result === 'fake' ? 'Fake' : 'Real'}</h3>}
        </div>
    );
};

export default FileUpload;
