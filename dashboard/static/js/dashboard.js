// Enhanced Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.dashboard-section');
    
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all nav items and sections
            navItems.forEach(nav => nav.classList.remove('active'));
            sections.forEach(section => section.classList.remove('active'));
            
            // Add active class to clicked nav item
            this.classList.add('active');
            
            // Show corresponding section
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.add('active');
            }
        });
    });
    
    // Initialize charts
    initializeCharts();
    
    // Auto-refresh data
    setInterval(refreshData, 30000); // Refresh every 30 seconds
});

function initializeCharts() {
    // GPU Performance Chart
    const gpuCtx = document.getElementById('gpuChart').getContext('2d');
    new Chart(gpuCtx, {
        type: 'line',
        data: {
            labels: ['1m', '2m', '3m', '4m', '5m', '6m'],
            datasets: [{
                label: 'GPU Usage %',
                data: [25, 30, 35, 28, 32, 30],
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
    
    // Compression Chart
    const compressionCtx = document.getElementById('compressionChart').getContext('2d');
    new Chart(compressionCtx, {
        type: 'bar',
        data: {
            labels: ['Huffman', 'LZW', 'Arithmetic', 'Wavelet'],
            datasets: [{
                label: 'Compression Ratio',
                data: [1.99, 3.70, 1475.00, 5.00],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.8)',
                    'rgba(46, 204, 113, 0.8)',
                    'rgba(231, 76, 60, 0.8)',
                    'rgba(155, 89, 182, 0.8)'
                ],
                borderColor: [
                    '#3498db',
                    '#2ecc71',
                    '#e74c3c',
                    '#9b59b6'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    type: 'logarithmic'
                }
            }
        }
    });
}

function refreshData() {
    // Simulate data refresh
    console.log('Refreshing dashboard data...');
    
    // Update status indicators
    const statusDots = document.querySelectorAll('.status-dot');
    statusDots.forEach(dot => {
        dot.style.animation = 'none';
        setTimeout(() => {
            dot.style.animation = 'pulse 2s infinite';
        }, 100);
    });
    
    // Update metrics (simulate)
    const metricValues = document.querySelectorAll('.metric-value');
    metricValues.forEach(value => {
        if (value.textContent.includes('GB')) {
            const currentValue = parseFloat(value.textContent);
            const newValue = (currentValue + Math.random() * 0.1).toFixed(1);
            value.textContent = newValue + 'GB';
        }
    });
}

// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add keyboard navigation
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case '1':
                e.preventDefault();
                document.querySelector('a[href="#overview"]').click();
                break;
            case '2':
                e.preventDefault();
                document.querySelector('a[href="#services"]').click();
                break;
            case '3':
                e.preventDefault();
                document.querySelector('a[href="#gpu"]').click();
                break;
            case '4':
                e.preventDefault();
                document.querySelector('a[href="#compression"]').click();
                break;
            case '5':
                e.preventDefault();
                document.querySelector('a[href="#monitoring"]').click();
                break;
            case '6':
                e.preventDefault();
                document.querySelector('a[href="#logs"]').click();
                break;
        }
    }
});