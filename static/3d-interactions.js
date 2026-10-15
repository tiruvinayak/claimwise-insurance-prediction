/**
 * ClaimWise — Subtle 3D and Anti-Gravity Micro-Interactions
 * Inspired by CartIQ's clean presentation with ClaimWise light aesthetics.
 * Respects prefers-reduced-motion and provides static fallbacks.
 */

(function () {
    'use strict';

    // Check prefers-reduced-motion
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (motionQuery.matches) {
        return; // Respect accessibility; keep all transforms static
    }

    // 1. Subtle 3D Perspective Tilt on .tilt-card and .hero-info-card
    function initTiltCards() {
        const cards = document.querySelectorAll('.tilt-card, .hero-info-card, .model-card-best');

        cards.forEach(card => {
            let rafId = null;

            card.addEventListener('mouseenter', () => {
                card.style.transition = 'transform 0.15s ease-out, box-shadow 0.2s ease-out';
            });

            card.addEventListener('mousemove', (e) => {
                if (rafId) cancelAnimationFrame(rafId);
                rafId = requestAnimationFrame(() => {
                    const rect = card.getBoundingClientRect();
                    const x = (e.clientX - rect.left) / rect.width - 0.5;
                    const y = (e.clientY - rect.top) / rect.height - 0.5;

                    // Restrain maximum rotation to ±3.5 degrees for an elegant, professional feel
                    const rotX = (-y * 6).toFixed(2);
                    const rotY = (x * 6).toFixed(2);

                    card.style.transform = `perspective(1000px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(-4px) scale3d(1.008, 1.008, 1.008)`;
                });
            });

            card.addEventListener('mouseleave', () => {
                if (rafId) cancelAnimationFrame(rafId);
                card.style.transition = 'transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.35s ease';
                card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0) scale3d(1, 1, 1)';
            });
        });
    }

    // 2. Smooth Reveal for Cards on Scroll
    function initScrollReveal() {
        if (!('IntersectionObserver' in window)) return;

        const revealElements = document.querySelectorAll('.card, .step, .kpi-card, .timeline-v-item');
        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('cw-revealed');
                    obs.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -40px 0px'
        });

        revealElements.forEach(el => {
            el.classList.add('cw-reveal');
            observer.observe(el);
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            initTiltCards();
            initScrollReveal();
        });
    } else {
        initTiltCards();
        initScrollReveal();
    }
})();
