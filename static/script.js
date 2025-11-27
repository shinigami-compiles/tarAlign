/**
 * TarAlign - Interactive JavaScript (Updated for 20-Feature System)
 */

// Tab Switching
function openTab(evt, tabName) {
    const tabContents = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].style.display = "none";
    }

    const tabLinks = document.getElementsByClassName("tab-link");
    for (let i = 0; i < tabLinks.length; i++) {
        tabLinks[i].className = tabLinks[i].className.replace(" active", "");
    }

    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.style.display = "block";
    }

    if (evt && evt.currentTarget) {
        evt.currentTarget.className += " active";
    }
}

// File Upload Display
function updateFileName(input) {
    const display = document.getElementById('file-name-display');
    if (!display) return;
    
    if (input.files && input.files.length > 0) {
        const file = input.files[0];
        const fileName = file.name;
        const fileSize = (file.size / 1024).toFixed(2);
        
        display.innerHTML = `
            <strong>Selected:</strong> ${fileName} 
            <span style="opacity: 0.7;">(${fileSize} KB)</span>
        `;
        display.style.display = 'block';
        display.style.animation = 'fadeIn 0.3s ease';
    } else {
        display.style.display = 'none';
    }
}

// Reset Form to Default Values
function resetForm() {
    const inputs = document.querySelectorAll('.feature-input');
    inputs.forEach(input => {
        // Get default value from data attribute or use 0
        const defaultValue = input.getAttribute('value') || '0.0';
        input.value = defaultValue;
    });
    
    showNotification('All fields reset to default values', 'info');
}

// Show Notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#4fc3f7' : type === 'error' ? '#ef5350' : '#64b5f6'};
        color: #000;
        border-radius: 6px;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease;
        font-family: 'Rajdhani', sans-serif;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Form Validation (Enhanced for 20 Features)
function validateManualForm(event) {
    const form = event.target;
    const inputs = form.querySelectorAll('.feature-input');
    
    let hasError = false;
    let emptyCount = 0;
    
    inputs.forEach(input => {
        const value = input.value.trim();
        
        if (value === '') {
            emptyCount++;
            input.style.borderColor = '#ef5350';
            hasError = true;
        } else if (isNaN(value)) {
            input.style.borderColor = '#ef5350';
            hasError = true;
        } else {
            input.style.borderColor = '';
        }
    });
    
    if (hasError) {
        event.preventDefault();
        showNotification(`Please fix ${emptyCount} invalid field(s)`, 'error');
        
        // Scroll to first error
        const firstError = form.querySelector('input[style*="border-color: rgb(239, 83, 80)"]');
        if (firstError) {
            firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
            firstError.focus();
        }
        return false;
    }
    
    return true;
}

// Auto-save to localStorage (Draft Mode)
function enableDraftMode() {
    const form = document.querySelector('.manual-form');
    if (!form) return;
    
    const inputs = form.querySelectorAll('.feature-input');
    
    // Load saved draft on page load
    inputs.forEach(input => {
        const savedValue = localStorage.getItem(`draft_${input.id}`);
        if (savedValue !== null) {
            input.value = savedValue;
            input.style.borderColor = '#64b5f6'; // Indicate loaded from draft
        }
    });
    
    // Auto-save on input change
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            localStorage.setItem(`draft_${this.id}`, this.value);
            this.style.borderColor = '#64b5f6'; // Visual feedback
            
            setTimeout(() => {
                if (this.value === localStorage.getItem(`draft_${this.id}`)) {
                    this.style.borderColor = '';
                }
            }, 500);
        });
    });
    
    // Clear draft on successful submission
    form.addEventListener('submit', function(e) {
        if (!validateManualForm(e)) {
            return;
        }
        
        inputs.forEach(input => {
            localStorage.removeItem(`draft_${input.id}`);
        });
    });
    
    // Add "Load Draft" indicator if draft exists
    const hasDraft = Array.from(inputs).some(input => 
        localStorage.getItem(`draft_${input.id}`) !== null
    );
    
    if (hasDraft) {
        const draftNotice = document.createElement('div');
        draftNotice.className = 'draft-notice';
        draftNotice.innerHTML = `
            <span style="color: #64b5f6;">📝 Draft loaded from previous session</span>
            <button type="button" onclick="clearDraft()" style="margin-left: 1rem; padding: 0.3rem 0.8rem; background: transparent; border: 1px solid #64b5f6; color: #64b5f6; border-radius: 4px; cursor: pointer;">
                Clear Draft
            </button>
        `;
        form.insertBefore(draftNotice, form.firstChild);
    }
}

// Clear Draft
function clearDraft() {
    const inputs = document.querySelectorAll('.feature-input');
    inputs.forEach(input => {
        localStorage.removeItem(`draft_${input.id}`);
    });
    location.reload();
}

// Keyboard Shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + S to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            const submitBtn = document.querySelector('button[type="submit"]:not([disabled])');
            if (submitBtn) {
                submitBtn.click();
            }
        }
        
        // Ctrl/Cmd + R to reset (prevent browser refresh)
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
            e.preventDefault();
            resetForm();
        }
        
        // Escape to clear focus
        if (e.key === 'Escape') {
            document.activeElement.blur();
        }
    });
}

// Add Loading State to Submit Button
function addFormLoadingStates() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.disabled = true;
                const originalHTML = submitBtn.innerHTML;
                
                submitBtn.innerHTML = `
                    <div class="loader" style="width: 20px; height: 20px; border-width: 3px; margin: 0 auto; display: inline-block;"></div>
                    <span style="margin-left: 0.5rem;">Processing...</span>
                `;
                
                // Safety timeout
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalHTML;
                }, 30000);
            }
        });
    });
}

// Input Validation on Blur
function setupInputValidation() {
    const inputs = document.querySelectorAll('.feature-input');
    
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            const value = parseFloat(this.value);
            const name = this.name;
            
            // Validate ranges based on field type
            if (name.includes('_index') || name.includes('weight') || name.includes('is_weekend')) {
                // These should be 0-1
                if (value < 0 || value > 1) {
                    this.style.borderColor = '#ef5350';
                    showNotification(`${name} should be between 0 and 1`, 'error');
                }
            } else if (name === 'day_of_week') {
                // Should be 0-6
                if (value < 0 || value > 6) {
                    this.style.borderColor = '#ef5350';
                    showNotification('Day of week should be 0-6', 'error');
                }
            } else if (value < 0) {
                // Most metrics shouldn't be negative
                this.style.borderColor = '#ef5350';
                showNotification(`${name} should not be negative`, 'error');
            }
        });
        
        // Clear error on focus
        input.addEventListener('focus', function() {
            this.style.borderColor = '';
        });
    });
}

// Progress Indicator for 20 Questions
function addProgressIndicator() {
    const form = document.querySelector('.manual-form');
    if (!form) return;
    
    const inputs = form.querySelectorAll('.feature-input');
    const totalQuestions = inputs.length;
    
    // Create progress bar
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress-container';
    progressContainer.innerHTML = `
        <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: var(--text-secondary);">Progress</span>
                <span id="progress-text" style="color: var(--glacier-primary); font-weight: 700;">0/${totalQuestions}</span>
            </div>
            <div style="width: 100%; height: 8px; background: var(--bg-primary); border-radius: 4px; overflow: hidden;">
                <div id="progress-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, var(--glacier-primary), var(--glacier-accent)); transition: width 0.3s ease;"></div>
            </div>
        </div>
    `;
    
    const featureGrid = form.querySelector('.feature-grid-20');
    if (featureGrid) {
        form.insertBefore(progressContainer, featureGrid);
    }
    
    // Update progress
    function updateProgress() {
        const filledInputs = Array.from(inputs).filter(input => input.value.trim() !== '' && input.value !== '0' && input.value !== '0.0');
        const progress = (filledInputs.length / totalQuestions) * 100;
        
        document.getElementById('progress-bar').style.width = `${progress}%`;
        document.getElementById('progress-text').textContent = `${filledInputs.length}/${totalQuestions}`;
    }
    
    inputs.forEach(input => {
        input.addEventListener('input', updateProgress);
    });
    
    updateProgress(); // Initial update
}

// Initialize Everything
document.addEventListener("DOMContentLoaded", function() {
    console.log('🎯 TarAlign UI initialized (20-Feature System)');
    
    // Activate default tab
    const defaultTab = document.querySelector(".tab-link.active");
    if (defaultTab) {
        defaultTab.click();
    }
    
    // Setup all features
    setupKeyboardShortcuts();
    addFormLoadingStates();
    enableDraftMode();
    setupInputValidation();
    addProgressIndicator();
    
    // Add form validation
    const manualForm = document.querySelector('.manual-form');
    if (manualForm) {
        manualForm.addEventListener('submit', validateManualForm);
    }
    
    // Smooth scroll for Help page anchors
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    
    console.log('✓ All features loaded successfully');
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .draft-notice {
        background: rgba(100, 181, 246, 0.1);
        border: 1px solid #64b5f6;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
`;
document.head.appendChild(style);