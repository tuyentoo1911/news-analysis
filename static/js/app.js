// THYNK UNLIMITED - Navigation and UI Enhancement
document.addEventListener('DOMContentLoaded', function() {
    // Set active navigation pill based on current URL
    const currentPath = window.location.pathname;
    const pills = document.querySelectorAll('.pill');
    
    pills.forEach(pill => {
        pill.classList.remove('active');
        const href = pill.getAttribute('href');
        
        if (href === currentPath || 
            (currentPath === '/' && href.includes('home')) ||
            (currentPath.includes('analyze') && href.includes('analyze')) ||
            (currentPath.includes('stats') && href.includes('stats')) ||
            (currentPath.includes('about') && href.includes('about'))) {
            pill.classList.add('active');
        }
    });
    
    // Enhanced hover effects for pills
    pills.forEach((pill) => {
        pill.addEventListener('mouseenter', () => {
            if (!pill.classList.contains('active')) {
                pill.style.transform = 'translateY(-1px)';
                pill.style.boxShadow = '0 2px 8px rgba(10, 20, 66, 0.15)';
            }
        });
        
        pill.addEventListener('mouseleave', () => {
            if (!pill.classList.contains('active')) {
                pill.style.transform = '';
                pill.style.boxShadow = '';
            }
        });
    });
    
    // Smooth scroll for internal links
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
});

// Typewriter effect for hero title with erase and loop
(() => {
  const container = document.querySelector('.typewriter');
  if (!container) return;
  const speed = Number(container.getAttribute('data-speed') || 35);
  const deleteSpeed = Number(container.getAttribute('data-delete-speed') || Math.max(20, Math.round(speed * 0.6)));
  const startDelay = Number(container.getAttribute('data-delay') || 150);
  const lines = Array.from(container.querySelectorAll('.tw-line'));
  const loop = container.getAttribute('data-loop') === 'true';

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const makeCursor = (el) => { const c = document.createElement('span'); c.className = 'cursor'; el.after(c); return c; };

  async function typeLine(lineEl, text) {
    lineEl.textContent = '';
    const cursor = makeCursor(lineEl);
    for (let i = 1; i <= text.length; i++) {
      lineEl.textContent = text.slice(0, i);
      await sleep(speed);
    }
    cursor.remove();
  }

  async function deleteLine(lineEl) {
    const base = lineEl.textContent || lineEl.getAttribute('data-text') || '';
    const cursor = makeCursor(lineEl);
    for (let i = base.length; i >= 0; i--) {
      lineEl.textContent = base.slice(0, i);
      await sleep(deleteSpeed);
    }
    cursor.remove();
  }

  (async () => {
    await sleep(startDelay);
    do {
      for (const el of lines) {
        const text = el.getAttribute('data-text') || '';
        await typeLine(el, text);
      }
      await sleep(1000);
      for (let i = lines.length - 1; i >= 0; i--) {
        await deleteLine(lines[i]);
      }
      await sleep(400);
    } while (loop);
  })();
})();


