<template>
    <div class="row">
        <a :href="link" class="dark-link">
            <i class="bi bi-github icon-space"></i> {{ text }}
        </a>
    </div>
</template>

<script>
    export default {
        props: ["text", "link"],
    }
</script>