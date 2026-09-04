.PHONY: llama-up llama-down llama-status

llama-up:
	@cp /Users/rickdevmajumder/Downloads/Lang-learn-project/com.langlearn.llama-server.plist ~/Library/LaunchAgents/com.langlearn.llama-server.plist 2>/dev/null || true
	@launchctl load ~/Library/LaunchAgents/com.langlearn.llama-server.plist 2>/dev/null || true
	@launchctl start com.langlearn.llama-server
	@echo "llama-server starting on :9090 (Metal-accelerated, Gemma 4 E4B)..."
	@sleep 3
	@$(MAKE) llama-status

llama-down:
	@launchctl stop com.langlearn.llama-server 2>/dev/null || true
	@launchctl unload ~/Library/LaunchAgents/com.langlearn.llama-server.plist 2>/dev/null || true
	@pkill -f "llama-server.*9090" 2>/dev/null || true
	@echo "llama-server stopped"

llama-status:
	@echo "Checking llama-server on http://localhost:9090/health..."
	@curl -s http://localhost:9090/health || echo "NOT RUNNING"

argocd-ui:
	@echo "Opening ArgoCD port-forward on https://localhost:8080..."
	@kubectl port-forward svc/argocd-server -n argocd 8080:443 --address 0.0.0.0

argocd-pass:
	@echo -n "ArgoCD admin password: "
	@kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo ""
