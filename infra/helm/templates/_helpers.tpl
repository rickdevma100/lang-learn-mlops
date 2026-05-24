{{/*
Reusable labels for all lang-learn resources.
*/}}
{{- define "lang-learn.labels" -}}
app.kubernetes.io/name: lang-learn
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
