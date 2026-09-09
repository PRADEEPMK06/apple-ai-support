# Intent Taxonomy — AppleSupport

## Intents

### 1. software_bug
Customer reports a specific bug introduced by an iOS/macOS update — broken features, glitches, autocorrect issues, display problems.
- Include: update broke something specific
- Exclude: general slowness without mentioning a bug

### 2. device_performance
Device is slow, freezing, lagging, or unresponsive. Not caused by a specific identifiable bug.
- Include: slow after update, phone freezing, safari slow
- Exclude: battery drain (that's battery_issue)

### 3. battery_issue
Battery draining unusually fast, incorrect percentage shown, charging taking too long, battery health questions.
- Include: battery drain, charging slow, percentage jumping
- Exclude: phone dying because of a bug (software_bug)

### 4. connectivity_issue
WiFi, Bluetooth, mobile data, or network not connecting or dropping frequently.
- Include: wifi forgetting password, no connection, bluetooth dropping
- Exclude: slow internet due to general performance (device_performance)

### 5. account_and_services
Apple ID setup, iCloud, App Store, iTunes, activation, subscription issues.
- Include: can't activate, Apple ID stuck, App Store not loading
- Exclude: app crashes (software_bug)

### 6. hardware_issue
Physical device defect — speaker not working, camera issue, screen problem, physical damage.
- Include: speaker dead, camera stuttering, screen glitch
- Exclude: software-caused display issues (software_bug)

### 7. general_question
Customer asking how to do something or requesting information, not reporting a problem.
- Include: how do I check battery health, what is the wait time
- Exclude: questions about a broken feature (software_bug)

### 8. complaint_feedback
Pure frustration or feedback with no specific actionable ask.
- Include: this is unacceptable, worst update ever, I'm switching
- Exclude: complaints that include a specific problem (classify by the problem)