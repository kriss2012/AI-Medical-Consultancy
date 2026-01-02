async function init() {
    try {
        const res = await fetch('/api/config');
        const config = await res.json();
        renderUI(config.user);
    } catch (e) { console.error(e); }
}

function renderUI(user) {
    const authDiv = document.getElementById('auth-section');
    const subContainer = document.getElementById('sub-btn-container');

    if (user) {
        // Logged In
        authDiv.innerHTML = `
            <div class="flex items-center gap-4">
                <span class="text-sm text-gray-300 hidden md:inline">${user.name}</span>
                <img src="${user.picture}" class="h-9 w-9 rounded-full border border-gray-600">
                ${user.is_admin ? '<a href="/admin" class="text-xs bg-gray-800 px-2 py-1 rounded text-white">Admin</a>' : ''}
                <a href="/auth/logout" class="text-sm text-red-400 hover:text-red-300">Logout</a>
            </div>
        `;

        if (user.is_pro) {
            subContainer.innerHTML = `<button disabled class="w-full bg-green-600/20 text-green-500 font-bold py-3 rounded-xl border border-green-500/50">Pro Active</button>`;
        } else {
            subContainer.innerHTML = `<button onclick="subscribe()" class="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:opacity-90 text-white font-bold py-3 rounded-xl transition shadow-lg">Upgrade - ₹499</button>`;
        }
    } else {
        // Logged Out
        authDiv.innerHTML = `
            <a href="/auth/login" class="bg-white text-gray-900 hover:bg-gray-100 px-5 py-2 rounded-lg font-medium transition shadow-lg">
                Sign In
            </a>
        `;
        subContainer.innerHTML = `<p class="text-sm text-center text-gray-500 italic">Sign in to upgrade</p>`;
    }
}

async function diagnose() {
    const symptoms = document.getElementById('symptoms').value;
    if (!symptoms) return alert("Please enter symptoms.");

    const btn = document.querySelector('button[onclick="diagnose()"]');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/diagnose', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symptoms})
        });

        if (res.status === 401) {
            alert("Please Sign In first.");
            window.location.href = '/auth/login';
            return;
        }

        const data = await res.json();
        
        document.getElementById('result-area').classList.remove('hidden');
        document.getElementById('diagnosis-text').innerText = data.primary_diagnosis;
        document.getElementById('confidence').innerText = (data.confidence * 100).toFixed(1) + "% Confidence";
        
        const list = document.getElementById('rec-list');
        list.innerHTML = '';
        data.recommendation.forEach(rec => {
            list.innerHTML += `<li>${rec}</li>`;
        });

    } catch (e) {
        alert("Error analyzing symptoms.");
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

async function subscribe() {
    try {
        const res = await fetch('/api/create_subscription', {method: 'POST'});
        const data = await res.json();

        const options = {
            key: data.key,
            amount: data.amount,
            currency: "INR",
            name: "MediAI Pro",
            description: "Pro Subscription",
            order_id: data.order_id,
            handler: async function (response) {
                await fetch('/api/verify_payment', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(response)
                });
                alert("Subscription Active!");
                location.reload();
            },
            theme: { color: "#6366f1" }
        };
        new Razorpay(options).open();
    } catch (e) {
        alert("Payment initialization failed.");
    }
}

init();