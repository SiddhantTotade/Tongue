import React, { useState } from 'react';

const languages = [
    { name: 'Hindi', code: 'hi' },
    { name: 'French', code: 'fr' },
    { name: 'Spanish', code: 'es' },
    { name: 'German', code: 'de' }
];

const Popup = () => {
    const [selectedLang, setSelectedLang] = useState('hi');

    const handleLanguageChange = (langCode) => {
        setSelectedLang(langCode);

        // Send message to content.js in the active YouTube tab
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, {
                type: "CHANGE_LANGUAGE",
                lang: langCode
            });
        });
    };

    return (
        <div style={{ padding: '20px', width: '250px', fontFamily: 'Arial' }}>
            <h3>Tongue Translator</h3>

            <label>Target Language:</label>
            <select
                value={selectedLang}
                onChange={(e) => handleLanguageChange(e.target.value)}
                style={{ width: '100%', padding: '8px', marginTop: '10px' }}
            >
                {languages.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                        {lang.name}
                    </option>
                ))}
            </select>

            <div style={{ marginTop: '20px', fontSize: '12px', color: '#666' }}>
                Status: Connected to OMEN Backend
            </div>
        </div>
    );
};

import { createRoot } from 'react-dom/client';
const container = document.getElementById('root');
const root = createRoot(container);
root.render(<Popup />);