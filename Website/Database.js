// ============================================================
// database.js
// Basic Firestore helper functions for the face game.
// Each level calls saveGameResult() with its level number.
//
// HOW TO SET UP FIREBASE:
//   1. Go to https://console.firebase.google.com
//   2. Create a project and enable Firestore Database
//   3. Replace the firebaseConfig values below with your own
//      (found in Project Settings > Your apps > SDK setup)
// ============================================================

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getFirestore, doc, setDoc } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

// ---- REPLACE THESE VALUES WITH YOUR OWN FIREBASE CONFIG ----
const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
};
// ------------------------------------------------------------

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// ============================================================
// saveGameResult()
//
// Saves the score for one level to Firestore.
// Called by game1.html when the player clicks Next on each level.
//
// Parameters:
//   userId   - string, e.g. "player_001" or "anonymous"
//   level    - number, 1 to 4
//   score    - number, 0 to 100
//   extra    - object, any extra info you want to store (optional)
//
// Document ID in Firestore: userId__level1, userId__level2, etc.
// Collection name: "game_results"
// ============================================================
export async function saveGameResult({ userId, level, score, extra = {} }) {
    const docId = `${userId}__level${level}`;
    const data = {
        userId,
        level,
        score,
        savedAt: new Date().toISOString(),
        ...extra
    };

    // setDoc with merge: true means it will update if doc already exists
    await setDoc(doc(db, "game_results", docId), data, { merge: true });
    console.log(`[database.js] Saved level ${level} score (${score}) for user: ${userId}`);
}