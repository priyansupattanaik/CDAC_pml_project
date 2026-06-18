document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // 1. Tab Navigation Routing
    // ----------------------------------------------------
    const btnEda = document.getElementById("btn-eda");
    const btnPredict = document.getElementById("btn-predict");
    const btnModel = document.getElementById("btn-model");

    const tabEda = document.getElementById("tab-eda-content");
    const tabPredict = document.getElementById("tab-predict-content");
    const tabModel = document.getElementById("tab-model-content");

    const navButtons = [btnEda, btnPredict, btnModel];
    const tabs = [tabEda, tabPredict, tabModel];

    function switchTab(index) {
        navButtons.forEach((btn, i) => {
            if (i === index) {
                btn.classList.add("active");
                tabs[i].classList.add("active");
            } else {
                btn.classList.remove("active");
                tabs[i].classList.remove("active");
            }
        });
    }

    btnEda.addEventListener("click", () => switchTab(0));
    btnPredict.addEventListener("click", () => switchTab(1));
    btnModel.addEventListener("click", () => switchTab(2));

    // ----------------------------------------------------
    // 2. Render Chart.js EDA Visualizations (100% Dynamic)
    // ----------------------------------------------------
    // Default Static Pre-calculated stats (for zero-latency fallback)
    const fallbackData = {
        attrition_distribution: { 'No': 1256, 'Yes': 396 },
        department_attrition: {
            'Human Resources': { 'No': 51, 'Yes': 26 },
            'Research & Development': { 'No': 846, 'Yes': 211 },
            'Sales': { 'No': 359, 'Yes': 159 }
        },
        overtime_attrition: {
            'No': { 'No': 962, 'Yes': 185 },
            'Yes': { 'No': 294, 'Yes': 211 }
        },
        marital_attrition: {
            'Divorced': { 'No': 302, 'Yes': 54 },
            'Married': { 'No': 599, 'Yes': 135 },
            'Single': { 'No': 355, 'Yes': 207 }
        },
        job_role_attrition: {
            'Healthcare Representative': { 'No': 127, 'Yes': 10 },
            'Human Resources': { 'No': 40, 'Yes': 26 },
            'Laboratory Technician': { 'No': 201, 'Yes': 91 },
            'Manager': { 'No': 100, 'Yes': 5 },
            'Manufacturing Director': { 'No': 138, 'Yes': 17 },
            'Research Director': { 'No': 78, 'Yes': 6 },
            'Research Scientist': { 'No': 250, 'Yes': 85 },
            'Sales Executive': { 'No': 272, 'Yes': 99 },
            'Sales Representative': { 'No': 50, 'Yes': 57 }
        },
        wlb_attrition: {
            '1.0': { 'No': 51, 'Yes': 40 },
            '2.0': { 'No': 267, 'Yes': 97 },
            '3.0': { 'No': 732, 'Yes': 182 },
            '4.0': { 'No': 109, 'Yes': 45 }
        }
    };

    function renderAllCharts(data) {
        // Chart.js Theme Defaults (Optimized for premium Light Mode)
        Chart.defaults.color = '#64748b';
        Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

        // Chart 1: Attrition Distribution
        const ctxAttrDist = document.getElementById("chart-attrition-dist").getContext("2d");
        new Chart(ctxAttrDist, {
            type: 'doughnut',
            data: {
                labels: ['Retention (No)', 'Attrition (Yes)'],
                datasets: [{
                    data: [data.attrition_distribution.No || 0, data.attrition_distribution.Yes || 0],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: '#ffffff',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { boxWidth: 15, padding: 15 }
                    }
                }
            }
        });

        // Chart 2: Attrition by Department
        const depts = ['Human Resources', 'Research & Development', 'Sales'];
        const ctxDept = document.getElementById("chart-department-attr").getContext("2d");
        new Chart(ctxDept, {
            type: 'bar',
            data: {
                labels: depts,
                datasets: [
                    {
                        label: 'No Attrition',
                        data: depts.map(d => (data.department_attrition[d] ? data.department_attrition[d].No : 0)),
                        backgroundColor: '#10b981',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    {
                        label: 'Attrition',
                        data: depts.map(d => (data.department_attrition[d] ? data.department_attrition[d].Yes : 0)),
                        backgroundColor: '#ef4444',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(0,0,0,0.05)' } }
                },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Chart 3: Attrition by Overtime
        const otKeys = ['No', 'Yes'];
        const ctxOvertime = document.getElementById("chart-overtime-attr").getContext("2d");
        new Chart(ctxOvertime, {
            type: 'bar',
            data: {
                labels: ['No Overtime', 'Overtime'],
                datasets: [
                    {
                        label: 'No Attrition',
                        data: otKeys.map(k => (data.overtime_attrition[k] ? data.overtime_attrition[k].No : 0)),
                        backgroundColor: '#10b981',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    {
                        label: 'Attrition',
                        data: otKeys.map(k => (data.overtime_attrition[k] ? data.overtime_attrition[k].Yes : 0)),
                        backgroundColor: '#ef4444',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(0,0,0,0.05)' } }
                },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Chart 4: Attrition by Marital Status
        const maritalKeys = ['Divorced', 'Married', 'Single'];
        const ctxMarital = document.getElementById("chart-marital-attr").getContext("2d");
        new Chart(ctxMarital, {
            type: 'bar',
            data: {
                labels: maritalKeys,
                datasets: [
                    {
                        label: 'No Attrition',
                        data: maritalKeys.map(k => (data.marital_attrition[k] ? data.marital_attrition[k].No : 0)),
                        backgroundColor: '#10b981',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    {
                        label: 'Attrition',
                        data: maritalKeys.map(k => (data.marital_attrition[k] ? data.marital_attrition[k].Yes : 0)),
                        backgroundColor: '#ef4444',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(0,0,0,0.05)' } }
                },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Chart 5: Attrition by Job Role
        const roles = [
            'Healthcare Representative', 'Human Resources', 'Laboratory Technician', 
            'Manager', 'Manufacturing Director', 'Research Director', 
            'Research Scientist', 'Sales Executive', 'Sales Representative'
        ];
        const ctxJobRole = document.getElementById("chart-job-role-attr").getContext("2d");
        new Chart(ctxJobRole, {
            type: 'bar',
            data: {
                labels: roles,
                datasets: [
                    {
                        label: 'No Attrition',
                        data: roles.map(r => (data.job_role_attrition[r] ? data.job_role_attrition[r].No : 0)),
                        backgroundColor: '#10b981',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    {
                        label: 'Attrition',
                        data: roles.map(r => (data.job_role_attrition[r] ? data.job_role_attrition[r].Yes : 0)),
                        backgroundColor: '#ef4444',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(0,0,0,0.05)' } },
                    y: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });

        // Chart 6: Attrition by Work Life Balance
        const wlbKeys = ['1.0', '2.0', '3.0', '4.0'];
        const ctxWlb = document.getElementById("chart-wlb-attr").getContext("2d");
        new Chart(ctxWlb, {
            type: 'bar',
            data: {
                labels: ['1.0 (Bad)', '2.0', '3.0', '4.0 (Best)'],
                datasets: [
                    {
                        label: 'No Attrition',
                        data: wlbKeys.map(k => (data.wlb_attrition[k] ? data.wlb_attrition[k].No : 0)),
                        backgroundColor: '#10b981',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    {
                        label: 'Attrition',
                        data: wlbKeys.map(k => (data.wlb_attrition[k] ? data.wlb_attrition[k].Yes : 0)),
                        backgroundColor: '#ef4444',
                        borderColor: '#ffffff',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(0,0,0,0.05)' } }
                },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    // Fetch dynamic aggregate statistics from backend, fallback if server is offline
    fetch('http://127.0.0.1:5000/api/eda')
        .then(res => res.json())
        .then(resData => {
            if (resData.success) {
                console.log("Successfully loaded live EDA statistics from CSV!");
                renderAllCharts(resData);
            } else {
                console.warn("API succeeded but returned error, using pre-calculated fallback charts.");
                renderAllCharts(fallbackData);
            }
        })
        .catch(err => {
            console.warn("Prediction server offline or /api/eda unreachable. Using zero-latency fallback charts.", err);
            renderAllCharts(fallbackData);
        });


    // ----------------------------------------------------
    // 3. Real-Time Prediction API Connection
    // ----------------------------------------------------
    const form = document.getElementById("prediction-form");
    const resultPanel = document.getElementById("result-panel");
    const riskBadge = document.getElementById("risk-badge");
    const progressCircle = document.getElementById("progress-circle");
    const progressVal = document.getElementById("progress-val");
    const recList = document.getElementById("rec-list");

    // Connect to check status on start
    fetch('http://127.0.0.1:5000/health')
        .then(response => response.json())
        .then(data => {
            console.log("Prediction backend connected! Status:", data.status);
            document.querySelector(".status-indicator").innerHTML = '<span class="dot"></span> API Server Connected';
        })
        .catch(err => {
            console.error("API backend connection error:", err);
            document.querySelector(".status-indicator").innerHTML = '<span class="dot" style="background-color: #ef4444; box-shadow: 0 0 8px #ef4444;"></span> API Server Disconnected';
        });

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        
        // Collate form fields
        const formData = new FormData(form);
        const payload = {};
        for (const [key, value] of formData.entries()) {
            payload[key] = value;
        }

        // Send POST request
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                // Show result panel
                resultPanel.style.display = "block";
                resultPanel.scrollIntoView({ behavior: 'smooth' });

                // Update risk badge
                if (data.prediction === 1) {
                    riskBadge.textContent = "HIGH RISK OF ATTRITION";
                    riskBadge.className = "risk-badge high";
                    document.querySelector(".recommendations-box").className = "recommendations-box high-risk";
                } else {
                    riskBadge.textContent = "LOW RISK (RETENTION)";
                    riskBadge.className = "risk-badge low";
                    document.querySelector(".recommendations-box").className = "recommendations-box";
                }

                // Update Progress Circle Gauge
                const probPercent = Math.round(data.probability * 100);
                progressVal.textContent = `${probPercent}%`;
                
                // Color conic gradient based on class
                const trackColor = data.prediction === 1 ? '#ef4444' : '#2E5BFF';
                progressCircle.style.background = `conic-gradient(${trackColor} ${probPercent * 3.6}deg, #f1f5f9 0deg)`;

                // Update Recommendations
                recList.innerHTML = "";
                data.recommendations.forEach(rec => {
                    const li = document.createElement("li");
                    li.textContent = rec;
                    recList.appendChild(li);
                });
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(err => {
            console.error("Prediction error:", err);
            alert("Could not connect to prediction server. Please make sure server.py is running on port 5000.");
        });
    });

    // ----------------------------------------------------
    // 4. Random Profile Generator
    // ----------------------------------------------------
    const btnRandom = document.getElementById("btn-random");
    if (btnRandom) {
        btnRandom.addEventListener("click", () => {
            // Randomize numeric range inputs (sliders)
            const rangeInputs = form.querySelectorAll('input[type="range"]');
            rangeInputs.forEach(input => {
                const min = parseInt(input.min) || 0;
                const max = parseInt(input.max) || 100;
                // Generate random value
                let randVal = Math.floor(Math.random() * (max - min + 1)) + min;
                
                // If monthly income, let's step it by 100
                if (input.id === 'MonthlyIncome') {
                    randVal = Math.round(randVal / 100) * 100;
                }
                
                input.value = randVal;
                
                // Update the span indicator text next to the label
                const valSpan = document.getElementById(`val-${input.id}`);
                if (valSpan) {
                    valSpan.textContent = randVal;
                }
            });

            // Randomize select dropdown elements
            const selectInputs = form.querySelectorAll('select');
            selectInputs.forEach(select => {
                const options = select.options;
                const randIndex = Math.floor(Math.random() * options.length);
                select.selectedIndex = randIndex;
            });
            
            console.log("Random profile generated successfully!");
        });
    }
});
